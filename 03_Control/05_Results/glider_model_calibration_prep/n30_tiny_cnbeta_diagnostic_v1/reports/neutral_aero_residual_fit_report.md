# Neutral Aero Residual Regime Fit

This run uses only neutral open-loop real throws. It estimates 6-DoF force/moment residuals from Vicon state trajectories, fits a claim-bearing longitudinal candidate by default, reports lateral residuals without claiming accurate lateral SysID, and validates candidates by held-out dry-air replay. The default `cm_regime_staged` workflow fits attached Cm, transition Cm, post-stall Cm/Cmq, transition blend, and optional post-stall CL/CD cleanup in separate held-out-gated stages. The `compact_joint_sweep` workflow starts from active constants, keeps the same compact model family, and jointly sweeps longitudinal plus small lateral/coupling terms after sign/range discovery. Longitudinal fitting uses lateral-contamination confidence; lateral/coupling fitting uses excitation-aware confidence. Rich transition lateral deltas, post-stall lateral surfaces, and post-stall alpha-RBF longitudinal surfaces are diagnostic-only unless explicitly enabled.

## Rerun Recipe

- source session root: `C:\Users\GH-X-ST\OneDrive - Imperial College London\Year 4\Final Year Project\01 - Github\Nausicaa\04_Flight_Test\05_Results\cal\n30`
- alignment window: `0.100` s
- derivative window: `0.040` s
- replay dt: `0.0050` s
- ridge lambda: `0.001`
- min speed: `1.50` m/s
- workers: `8`
- fit workflow: `residual_only`
- group iterations: `3`
- group improvement tolerance: `0.001`
- aligned launch filter: `True`
- aligned launch filter bounds: `u=[3.00, 8.00]` m/s, `|v|<=1.50` m/s, `|w|<=0.90` m/s
- apply attached Cm bias: `False`
- fit transition Cm bias: `False`
- fit post-stall CL/CD cleanup: `False`
- fit post-stall Cm bias: `False`
- fit compact post-stall longitudinal residuals: `False`
- fit transition blender: `False`
- fit post-stall alpha-RBF surfaces: `False`
- fit post-stall damping: `False`
- fit attached lateral coupling: `False`
- fit transition lateral coupling: `False`
- fit lateral surfaces: `False`
- fit secondary lateral diagnostic: `False`

```powershell
python 03_Control/02_Inner_Loop/run_fit_neutral_aero_residual_calibration.py --session-root "C:\Users\GH-X-ST\OneDrive - Imperial College London\Year 4\Final Year Project\01 - Github\Nausicaa\04_Flight_Test\05_Results\cal\n30" --run-label n30_tiny_cnbeta_diagnostic_v1 --heldout-count 11 --heldout-seed 606 --alignment-window-s 0.1 --derivative-window-s 0.04 --replay-dt-s 0.005 --ridge-lambda 0.001 --min-speed-m-s 1.5 --workers 8 --fit-workflow residual_only --group-iterations 3 --group-improvement-tol 0.001 --aligned-u-min-m-s 3 --aligned-u-max-m-s 8 --aligned-v-abs-max-m-s 1.5 --aligned-w-abs-max-m-s 0.9 --filter-aligned-launch-state --no-apply-attached-cm-bias --no-fit-transition-pitch-moment --no-fit-post-stall-lift-drag --no-fit-post-stall-pitch-moment --no-fit-post-stall-longitudinal --no-fit-transition-blender --no-fit-post-stall-surfaces --no-fit-post-stall-damping --no-fit-attached-lateral-coupling --no-fit-transition-lateral-coupling --no-fit-lateral-surfaces --no-fit-secondary-lateral-diagnostic
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
- fit MAE in Cm: `0.03810`
- attached Cm residual: `-0.0376994`
- transition Cm residual before post-stall: `-0.0456196`
- transition Cm residual after post-stall: `-0.0084878`
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
- post-stall Cmq residual: `0`
- selected compact post-stall replay scale: `0.000`
- transition blender status: `ok`
- transition blender fit group: `transition_before_post_stall`
- transition blender start alpha: `11.000` deg
- transition blender full alpha: `19.000` deg

## Grouped Replay Refinement

- grouped history rows: `0`
- grouped history CSV: `metrics/neutral_aero_residual_group_iteration_history.csv`
- grouped replay refinement disabled

## Regime-Staged Cm Workflow

- staged history rows: `0`
- staged history CSV: `metrics/neutral_aero_residual_cm_stage_history.csv`
- regime-staged Cm workflow disabled

## Secondary Lateral Diagnostic

- enabled: `False`
- status: `disabled`
- diagnostic coefficients CSV: `metrics/neutral_aero_residual_lateral_diagnostic_coefficients.csv`
- diagnostic policy: longitudinal parameters are frozen at the primary candidate; only `CY_beta`, `Cl_p`, and `Cn_r` may change; lateral-only fitting uses excitation-aware confidence weighting
- attached lateral coupling:
  - CY: bias `0`, beta `0`, p_hat `0`, r_hat `0`
  - Cl: bias `0`, beta `0`, p_hat `0`, r_hat `0`
  - Cn: bias `0`, beta `0`, p_hat `0`, r_hat `0`
- transition-window lateral coupling:
  - CY: bias `0`, beta `0`, p_hat `0`, r_hat `0`
  - Cl: bias `0`, beta `0`, p_hat `0`, r_hat `0`
  - Cn: bias `0`, beta `0`, p_hat `0`, r_hat `0`
- lateral diagnostic held-out dy/roll/yaw MAE: `inf` m, `inf` deg, `inf` deg
- lateral diagnostic acceptance: `rejected_diagnostic_only`
- lateral diagnostic policy: secondary lateral diagnostic is reportable only if held-out dy, roll, and yaw improve versus the primary longitudinal candidate while dx, altitude loss, sink, and pitch do not degrade
  - dy_mae_m (lateral_improvement_required): primary `0.5328`, diagnostic `nan`, delta `nan`, pass `False`
  - final_phi_mae_deg (lateral_improvement_required): primary `18.8764`, diagnostic `nan`, delta `nan`, pass `False`
  - final_psi_mae_deg (lateral_improvement_required): primary `6.7171`, diagnostic `nan`, delta `nan`, pass `False`
  - dx_mae_m (longitudinal_preservation_required): primary `0.3708`, diagnostic `nan`, delta `nan`, pass `False`
  - altitude_loss_mae_m (longitudinal_preservation_required): primary `0.1445`, diagnostic `nan`, delta `nan`, pass `False`
  - sink_mae_m_s (longitudinal_preservation_required): primary `0.1318`, diagnostic `nan`, delta `nan`, pass `False`
  - final_theta_mae_deg (longitudinal_preservation_required): primary `4.0439`, diagnostic `nan`, delta `nan`, pass `False`

## Lateral One-Term Ablation

- ablation CSV: `metrics/neutral_aero_residual_lateral_ablation.csv`
- policy: longitudinal candidate is frozen; each diagnostic candidate fits one lateral term and one regime family at a time. If held-out `attached_CY_beta` is accepted, a second pass tests remaining cross-couplings against that side-force baseline.
- acceptance: held-out dy, roll, and yaw must all improve while dx, altitude loss, sink, and pitch stay within practical tolerance.
- held-out ablations tested: `27`
- accepted held-out lateral ablations:
  - `transition_Cn_beta` vs `primary_longitudinal` coeff `-0.0146758` (negative); dy `0.533` -> `0.527` m, roll `18.88` -> `18.55` deg, yaw `6.72` -> `6.64` deg
  - `post_stall_Cn_beta` vs `primary_longitudinal` coeff `-0.0374172` (negative); dy `0.533` -> `0.502` m, roll `18.88` -> `16.28` deg, yaw `6.72` -> `6.44` deg
- best held-out dy reductions, even if rejected:
  - `attached_Cn_beta` vs `primary_longitudinal` coeff `-0.048748` delta dy `-0.118` m, delta roll `-5.82` deg, delta yaw `2.50` deg, reason `rejected_lateral_metrics_not_all_improved`
  - `post_stall_CY_r` vs `primary_longitudinal` coeff `-4` delta dy `-0.087` m, delta roll `-0.74` deg, delta yaw `-1.32` deg, reason `rejected_longitudinal_metrics_degraded`
  - `post_stall_Cn_beta` vs `primary_longitudinal` coeff `-0.0374172` delta dy `-0.031` m, delta roll `-2.59` deg, delta yaw `-0.28` deg, reason `heldout_lateral_improved_with_longitudinal_tolerance`
  - `post_stall_Cl_beta` vs `primary_longitudinal` coeff `0.102097` delta dy `-0.017` m, delta roll `7.37` deg, delta yaw `0.04` deg, reason `rejected_lateral_metrics_not_all_improved`
  - `post_stall_Cl_p` vs `primary_longitudinal` coeff `0.543932` delta dy `-0.011` m, delta roll `11.93` deg, delta yaw `-1.39` deg, reason `rejected_lateral_metrics_not_all_improved`

## Compact Joint Sweep

- candidate CSV: `metrics/neutral_aero_residual_joint_sweep_candidates.csv`
- pareto CSV: `metrics/neutral_aero_residual_joint_sweep_pareto.csv`
- selected CSV: `metrics/neutral_aero_residual_joint_sweep_selected.csv`
- policy: from active constants only; signs/ranges are discovered from current residuals, then compact longitudinal/lateral terms are swept jointly with 8-worker replay.
- compact joint sweep not run for this workflow

## Lateral Launch-Correlation Audit

- correlation CSV: `metrics/neutral_aero_residual_lateral_launch_correlation.csv`
- interpretation: strong correlation means the remaining lateral replay error is largely launch-condition dependent, so bad lateral launches should be down-weighted before stronger lateral aerodynamics are promoted.
- strongest held-out correlations for the accepted longitudinal candidate:
  - `roll` residual vs `p0_rad_s`: r `-0.767`, slope `-45.665`, n `11`
  - `dy` residual vs `phi0_deg`: r `-0.748`, slope `-0.096`, n `11`
  - `dy` residual vs `psi0_deg`: r `-0.623`, slope `-0.098`, n `11`
  - `roll` residual vs `psi0_deg`: r `-0.544`, slope `-1.885`, n `11`
  - `roll` residual vs `phi0_deg`: r `-0.474`, slope `-1.340`, n `11`
  - `yaw` residual vs `p0_rad_s`: r `-0.443`, slope `-16.648`, n `11`
  - `yaw` residual vs `r0_rad_s`: r `-0.419`, slope `-19.280`, n `11`
  - `yaw` residual vs `psi0_deg`: r `-0.360`, slope `-0.788`, n `11`
- interpretation: at least one held-out lateral residual has strong launch-condition correlation; down-weighting contaminated launches is likely safer than promoting stronger lateral aerodynamics.

## Replay Validation

- baseline train pitch MAE: `6.318` deg
- candidate train pitch MAE: `6.318` deg
- baseline held-out pitch MAE: `4.044` deg
- candidate held-out pitch MAE: `4.044` deg
- baseline held-out altitude-loss MAE: `0.1445` m
- candidate held-out altitude-loss MAE: `0.1445` m
- baseline held-out dx MAE: `0.3708` m
- candidate held-out dx MAE: `0.3708` m
- held-out acceptance: `accepted`
- acceptance policy: primary longitudinal candidate must improve or preserve held-out dx, altitude loss, sink, and pitch MAE versus baseline_active; lateral errors are reported but not claim-bearing
  - dx_mae_m: baseline `0.3708`, candidate `0.3708`, delta `0.0000`, pass `True`
  - altitude_loss_mae_m: baseline `0.1445`, candidate `0.1445`, delta `0.0000`, pass `True`
  - sink_mae_m_s: baseline `0.1318`, candidate `0.1318`, delta `0.0000`, pass `True`
  - final_theta_mae_deg: baseline `4.0439`, candidate `4.0439`, delta `0.0000`, pass `True`

## Stage Replay Errors

These are sample-aligned replay residuals grouped by the measured Vicon alpha regime. Use them to see whether the candidate fixes attached, transition, or post-stall behaviour; keep the full-throw held-out replay gate as the acceptance criterion.

- train/attached: samples `2248`, throws `63`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.107` m, `0.205` m, `0.084` m, `0.171` m/s, `8.71` deg, `4.05` deg, `3.20` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.107` m, `0.205` m, `0.084` m, `0.171` m/s, `8.71` deg, `4.05` deg, `3.20` deg
- train/transition: samples `7358`, throws `63`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.122` m, `0.218` m, `0.119` m, `0.210` m/s, `12.18` deg, `5.97` deg, `4.59` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.122` m, `0.218` m, `0.119` m, `0.210` m/s, `12.18` deg, `5.97` deg, `4.59` deg
- train/post_stall: samples `891`, throws `19`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.103` m, `0.428` m, `0.079` m, `0.115` m/s, `14.92` deg, `5.41` deg, `7.11` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.103` m, `0.428` m, `0.079` m, `0.115` m/s, `14.92` deg, `5.41` deg, `7.11` deg
- heldout/attached: samples `394`, throws `11`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.128` m, `0.162` m, `0.049` m, `0.175` m/s, `6.14` deg, `3.62` deg, `2.24` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.128` m, `0.162` m, `0.049` m, `0.175` m/s, `6.14` deg, `3.62` deg, `2.24` deg
- heldout/transition: samples `1260`, throws `11`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.143` m, `0.235` m, `0.111` m, `0.230` m/s, `12.83` deg, `7.43` deg, `4.64` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.143` m, `0.235` m, `0.111` m, `0.230` m/s, `12.83` deg, `7.43` deg, `4.64` deg
- heldout/post_stall: samples `251`, throws `5`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.283` m, `0.297` m, `0.209` m, `0.330` m/s, `18.22` deg, `9.26` deg, `6.26` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.283` m, `0.297` m, `0.209` m, `0.330` m/s, `18.22` deg, `9.26` deg, `6.26` deg

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
- post-stall Cn surface: bias: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`; beta: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`; p_hat: 20 deg baseline `-0.0711992` -> candidate `-0.0711992`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`; r_hat: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`
- attached lateral coupling: unchanged
- transition lateral coupling: unchanged
- baseline post-stall Cmq: `4`
- candidate post-stall Cmq: `4`
- baseline residual blend start alpha: `12.000` deg
- candidate residual blend start alpha: `12.000` deg
- baseline residual blend full alpha: `22.000` deg
- candidate residual blend full alpha: `22.000` deg

## Regime Summary

- train/attached: count `2311`, Cm mean `-0.04112`, Cm MAE `0.06000`, CY mean `-0.08179`, Cl mean `0.00905`, Cn mean `-0.00440`
- train/transition: count `7358`, Cm mean `-0.04057`, Cm MAE `0.04810`, CY mean `-0.15981`, Cl mean `-0.00868`, Cn mean `-0.00120`
- train/post_stall: count `891`, Cm mean `0.00081`, Cm MAE `0.04002`, CY mean `-0.49739`, Cl mean `-0.02329`, Cn mean `0.00585`
- heldout/attached: count `405`, Cm mean `-0.04139`, Cm MAE `0.06655`, CY mean `-0.12271`, Cl mean `0.01051`, Cn mean `-0.00242`
- heldout/transition: count `1260`, Cm mean `-0.03708`, Cm MAE `0.04958`, CY mean `-0.22499`, Cl mean `-0.01061`, Cn mean `0.00167`
- heldout/post_stall: count `251`, Cm mean `-0.02105`, Cm MAE `0.03752`, CY mean `-0.50038`, Cl mean `-0.03519`, Cn mean `-0.00533`

## Independent Stage Fit Summary

Transition is split into samples before and after the first post-stall exposure in each throw. This separates real transition-model evidence from transition samples that may already be contaminated by earlier post-stall divergence.

- train/attached: count `2311`, Cm mean `-0.04112`, Cm MAE `0.06000`, Cm fit residual MAE `0.04715`, CY mean `-0.08179`, Cl mean `0.00905`, Cn mean `-0.00440`
- train/transition_before_post_stall: count `6520`, Cm mean `-0.04392`, Cm MAE `0.04980`, Cm fit residual MAE `0.03457`, CY mean `-0.15306`, Cl mean `-0.00916`, Cn mean `-0.00132`
- train/transition_after_post_stall: count `838`, Cm mean `-0.01455`, Cm MAE `0.03487`, Cm fit residual MAE `0.03697`, CY mean `-0.21228`, Cl mean `-0.00494`, Cn mean `-0.00025`
- train/post_stall: count `891`, Cm mean `0.00081`, Cm MAE `0.04002`, Cm fit residual MAE `0.04152`, CY mean `-0.49739`, Cl mean `-0.02329`, Cn mean `0.00585`
- heldout/attached: count `405`, Cm mean `-0.04139`, Cm MAE `0.06655`, Cm fit residual MAE `0.05360`, CY mean `-0.12271`, Cl mean `0.01051`, Cn mean `-0.00242`
- heldout/transition_before_post_stall: count `1077`, Cm mean `-0.04314`, Cm MAE `0.04918`, Cm fit residual MAE `0.03554`, CY mean `-0.17088`, Cl mean `-0.01269`, Cn mean `-0.00123`
- heldout/transition_after_post_stall: count `183`, Cm mean `-0.00141`, Cm MAE `0.05193`, Cm fit residual MAE `0.05214`, CY mean `-0.54345`, Cl mean `0.00158`, Cn mean `0.01871`
- heldout/post_stall: count `251`, Cm mean `-0.02105`, Cm MAE `0.03752`, Cm fit residual MAE `0.03473`, CY mean `-0.50038`, Cl mean `-0.03519`, Cn mean `-0.00533`

## Interpretation

Accept the primary candidate only for the longitudinal claim-bearing model when held-out dx, altitude loss, sink, and pitch improve or preserve the active baseline. Treat dy, roll, and yaw as reported residual evidence unless the secondary lateral diagnostic improves held-out dy/roll/yaw without damaging those longitudinal metrics.
