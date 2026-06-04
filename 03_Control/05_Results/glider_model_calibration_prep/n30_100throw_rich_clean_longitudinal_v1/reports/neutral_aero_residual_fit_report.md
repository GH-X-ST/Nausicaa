# Neutral Aero Residual Regime Fit

This run uses only neutral open-loop real throws. It estimates 6-DoF force/moment residuals from Vicon state trajectories, fits a claim-bearing longitudinal candidate by default, reports lateral residuals without claiming accurate lateral SysID, and validates the primary candidate by held-out dry-air replay. The optional secondary lateral diagnostic freezes the longitudinal candidate and fits only `CY_beta`, `Cl_p`, and `Cn_r`; rich transition lateral deltas, post-stall lateral surfaces, and post-stall alpha-RBF longitudinal surfaces are diagnostic-only unless explicitly enabled.

## Rerun Recipe

- source session root: `04_Flight_Test\05_Results\cal\n30`
- alignment window: `0.100` s
- derivative window: `0.040` s
- replay dt: `0.0050` s
- ridge lambda: `0.001`
- min speed: `1.50` m/s
- workers: `8`
- fit workflow: `grouped_iterative`
- group iterations: `5`
- group improvement tolerance: `0.001`
- aligned launch filter: `True`
- aligned launch filter bounds: `u=[4.00, 8.00]` m/s, `|v|<=1.50` m/s, `|w|<=0.90` m/s
- apply attached Cm bias: `False`
- fit post-stall alpha-RBF surfaces: `False`
- fit post-stall damping: `True`
- fit attached lateral coupling: `False`
- fit transition lateral coupling: `False`
- fit lateral surfaces: `False`
- fit secondary lateral diagnostic: `True`

```powershell
python 03_Control/02_Inner_Loop/run_fit_neutral_aero_residual_calibration.py --session-root 04_Flight_Test\05_Results\cal\n30 --run-label n30_100throw_rich_clean_longitudinal_v1 --heldout-count 10 --heldout-seed 606 --alignment-window-s 0.1 --derivative-window-s 0.04 --replay-dt-s 0.005 --ridge-lambda 0.001 --min-speed-m-s 1.5 --workers 8 --fit-workflow grouped_iterative --group-iterations 5 --group-improvement-tol 0.001 --aligned-u-min-m-s 4 --aligned-u-max-m-s 8 --aligned-v-abs-max-m-s 1.5 --aligned-w-abs-max-m-s 0.9 --filter-aligned-launch-state --no-apply-attached-cm-bias --no-fit-post-stall-surfaces --fit-post-stall-damping --no-fit-attached-lateral-coupling --no-fit-transition-lateral-coupling --no-fit-lateral-surfaces --fit-secondary-lateral-diagnostic
```

## Aligned Launch Filter

- loaded logged-valid throws: `100`
- kept throws after replay-start filter: `67`
- filtered throws: `33`
- filter audit CSV: `metrics/neutral_aero_residual_filtered_throws.csv`
- launch-confidence weighting: enabled for residual coefficient fitting
- confidence reference: replay-aligned lateral contamination `phi0=psi0=v0=p0=r0=0`; minimum weight `0.25`
- kept-throw confidence weight min/mean/max: `0.5270482418`, `0.7406632276`, `0.931118568`
- kept-throw lateral score min/mean/max: `0.2181263461`, `0.4434086928`, `0.6534335948`
- rejected logged-valid throws:
  - `20260604_185934/v001`: aligned_w_outside_launch_gate; u0 `4.543059555`, v0 `0.3501702742`, w0 `1.008779882` m/s
  - `20260604_185934/v006`: aligned_v_outside_launch_gate; u0 `5.623978617`, v0 `1.917732218`, w0 `0.5411629165` m/s
  - `20260604_185934/v008`: aligned_u_outside_launch_gate; u0 `0.9721909615`, v0 `0.2445794314`, w0 `0.1511750445` m/s
  - `20260604_185934/v009`: aligned_u_outside_launch_gate; u0 `3.677703651`, v0 `1.045884071`, w0 `0.4535392762` m/s
  - `20260604_185934/v010`: aligned_w_outside_launch_gate; u0 `5.149037042`, v0 `1.238403229`, w0 `1.018297939` m/s
  - `20260604_190407/v002`: aligned_w_outside_launch_gate; u0 `4.97722127`, v0 `0.7757127111`, w0 `0.9282428727` m/s
  - `20260604_190407/v005`: aligned_w_outside_launch_gate; u0 `5.929172756`, v0 `0.609645104`, w0 `0.9622552935` m/s
  - `20260604_191406/v003`: aligned_w_outside_launch_gate; u0 `4.947658717`, v0 `1.138830854`, w0 `1.137633454` m/s
  - `20260604_191406/v008`: aligned_v_outside_launch_gate; u0 `5.288296354`, v0 `1.521964301`, w0 `0.4222859752` m/s
  - `20260604_191406/v009`: aligned_w_outside_launch_gate; u0 `5.790927828`, v0 `0.8098222082`, w0 `1.061122725` m/s
  - `20260604_195702/v001`: aligned_v_outside_launch_gate; u0 `4.468433221`, v0 `1.652145915`, w0 `0.3445464762` m/s
  - `20260604_195702/v006`: aligned_v_outside_launch_gate; u0 `5.305858468`, v0 `1.50640443`, w0 `0.5582058265` m/s
  - `20260604_195702/v010`: aligned_u_outside_launch_gate; u0 `3.738084589`, v0 `0.8785730579`, w0 `0.776907954` m/s
  - `20260604_203516/v004`: aligned_v_outside_launch_gate; u0 `5.300256981`, v0 `1.502123524`, w0 `0.3933348901` m/s
  - `20260604_203516/v014`: aligned_v_outside_launch_gate; u0 `5.407679143`, v0 `1.532947734`, w0 `0.7194391593` m/s
  - `20260604_203516/v015`: aligned_u_outside_launch_gate; u0 `3.706525382`, v0 `1.074292224`, w0 `0.6079756946` m/s
  - `20260604_203516/v019`: aligned_w_outside_launch_gate; u0 `5.076755044`, v0 `0.4722628657`, w0 `0.927494506` m/s
  - `20260604_203516/v023`: aligned_u_outside_launch_gate; u0 `3.805802493`, v0 `0.8096885689`, w0 `0.3521862968` m/s
  - `20260604_203516/v025`: aligned_u_outside_launch_gate; u0 `3.392460762`, v0 `0.6601617734`, w0 `0.3600294328` m/s
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
- sample count: `9646`
- used sample count: `9646`
- post-stall used sample count: `905`
- post-stall fit profile: `compact_scalar_activation`
- fit MAE in Cm: `0.04334`
- attached Cm residual: `0.0753363`
- transition Cm residual before post-stall: `0.0363558`
- transition Cm residual after post-stall: `0.0519428`
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
- post-stall Cmq residual: `3.74687`
- selected compact post-stall replay scale: `1.000`
- transition blender status: `ok`
- transition blender fit group: `transition_before_post_stall`
- transition blender start alpha: `11.000` deg
- transition blender full alpha: `20.000` deg

## Grouped Replay Refinement

- grouped history rows: `19`
- grouped history CSV: `metrics/neutral_aero_residual_group_iteration_history.csv`
- selected grouped replay scales:
  - pass `0`, group `initial_grouped_candidate`, scale ``, objective `71.9753`
  - pass `1`, group `post_stall_longitudinal`, scale `1.25`, objective `71.2425`
  - pass `1`, group `transition_blender`, scale `1`, objective `71.2425`
  - pass `2`, group `post_stall_longitudinal`, scale `1.25`, objective `71.2425`
  - pass `2`, group `transition_blender`, scale `1`, objective `71.2425`

## Secondary Lateral Diagnostic

- enabled: `True`
- status: `ok`
- diagnostic coefficients CSV: `metrics/neutral_aero_residual_lateral_diagnostic_coefficients.csv`
- diagnostic policy: longitudinal parameters are frozen at the primary candidate; only `CY_beta`, `Cl_p`, and `Cn_r` may change; launch-confidence weighting is ignored for this lateral-only fit
- attached lateral coupling:
  - CY: bias `0`, beta `-1.31077`, p_hat `0`, r_hat `0`
  - Cl: bias `0`, beta `0`, p_hat `0.256969`, r_hat `0`
  - Cn: bias `0`, beta `0`, p_hat `0`, r_hat `0.286962`
- transition-window lateral coupling:
  - CY: bias `0`, beta `0`, p_hat `0`, r_hat `0`
  - Cl: bias `0`, beta `0`, p_hat `0`, r_hat `0`
  - Cn: bias `0`, beta `0`, p_hat `0`, r_hat `0`
- lateral diagnostic held-out dy/roll/yaw MAE: `1.4295` m, `17.794` deg, `97.326` deg
- lateral diagnostic acceptance: `rejected_diagnostic_only`
- lateral diagnostic policy: secondary lateral diagnostic is reportable only if held-out dy, roll, and yaw improve versus the primary longitudinal candidate while dx, altitude loss, sink, and pitch do not degrade
  - dy_mae_m (lateral_improvement_required): primary `1.4644`, diagnostic `1.4295`, delta `-0.0348`, pass `True`
  - final_phi_mae_deg (lateral_improvement_required): primary `17.5434`, diagnostic `17.7942`, delta `0.2508`, pass `False`
  - final_psi_mae_deg (lateral_improvement_required): primary `12.0645`, diagnostic `97.3265`, delta `85.2620`, pass `False`
  - dx_mae_m (longitudinal_preservation_required): primary `0.9413`, diagnostic `0.4948`, delta `-0.4465`, pass `True`
  - altitude_loss_mae_m (longitudinal_preservation_required): primary `1.2447`, diagnostic `0.9223`, delta `-0.3223`, pass `True`
  - sink_mae_m_s (longitudinal_preservation_required): primary `1.1106`, diagnostic `0.8219`, delta `-0.2887`, pass `True`
  - final_theta_mae_deg (longitudinal_preservation_required): primary `19.8189`, diagnostic `17.0667`, delta `-2.7522`, pass `True`

## Replay Validation

- baseline train pitch MAE: `22.511` deg
- candidate train pitch MAE: `21.173` deg
- baseline held-out pitch MAE: `21.007` deg
- candidate held-out pitch MAE: `19.819` deg
- baseline held-out altitude-loss MAE: `1.2741` m
- candidate held-out altitude-loss MAE: `1.2447` m
- baseline held-out dx MAE: `0.9664` m
- candidate held-out dx MAE: `0.9413` m
- held-out acceptance: `accepted`
- acceptance policy: primary longitudinal candidate must improve or preserve held-out dx, altitude loss, sink, and pitch MAE versus baseline_active; lateral errors are reported but not claim-bearing
  - dx_mae_m: baseline `0.9664`, candidate `0.9413`, delta `-0.0251`, pass `True`
  - altitude_loss_mae_m: baseline `1.2741`, candidate `1.2447`, delta `-0.0295`, pass `True`
  - sink_mae_m_s: baseline `1.1373`, candidate `1.1106`, delta `-0.0267`, pass `True`
  - final_theta_mae_deg: baseline `21.0070`, candidate `19.8189`, delta `-1.1881`, pass `True`

## Stage Replay Errors

These are sample-aligned replay residuals grouped by the measured Vicon alpha regime. Use them to see whether the candidate fixes attached, transition, or post-stall behaviour; keep the full-throw held-out replay gate as the acceptance criterion.

- train/attached: samples `2070`, throws `57`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.238` m, `0.404` m, `0.354` m, `0.475` m/s, `6.47` deg, `10.85` deg, `5.06` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.239` m, `0.405` m, `0.347` m, `0.468` m/s, `6.94` deg, `10.52` deg, `5.16` deg
- train/transition: samples `6603`, throws `57`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.319` m, `0.517` m, `0.478` m, `0.697` m/s, `8.35` deg, `19.81` deg, `8.39` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.314` m, `0.516` m, `0.473` m, `0.692` m/s, `8.37` deg, `19.42` deg, `8.63` deg
- train/post_stall: samples `916`, throws `19`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.543` m, `0.829` m, `0.493` m, `0.634` m/s, `11.21` deg, `19.69` deg, `8.52` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.530` m, `0.830` m, `0.490` m, `0.631` m/s, `11.09` deg, `18.03` deg, `9.00` deg
- heldout/attached: samples `357`, throws `10`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.260` m, `0.460` m, `0.291` m, `0.413` m/s, `6.24` deg, `9.50` deg, `2.40` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.260` m, `0.460` m, `0.291` m, `0.413` m/s, `6.24` deg, `9.50` deg, `2.40` deg
- heldout/transition: samples `1300`, throws `10`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.362` m, `0.630` m, `0.488` m, `0.696` m/s, `9.91` deg, `20.25` deg, `8.35` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.351` m, `0.629` m, `0.482` m, `0.690` m/s, `9.92` deg, `19.90` deg, `8.65` deg
- heldout/post_stall: samples `121`, throws `3`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.755` m, `0.949` m, `0.998` m, `1.205` m/s, `8.35` deg, `30.60` deg, `6.01` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.755` m, `0.949` m, `0.998` m, `1.205` m/s, `8.35` deg, `30.60` deg, `6.01` deg

## Candidate Parameters

- legacy scalar post-stall CL residual: baseline `0`, candidate `-0.0229803`
- legacy scalar post-stall CD residual: baseline `0`, candidate `0.0624768`
- legacy scalar post-stall Cm residual: baseline `0`, candidate `0.0903288`
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
- candidate post-stall Cmq: `4`
- baseline residual blend start alpha: `12.000` deg
- candidate residual blend start alpha: `11.000` deg
- baseline residual blend full alpha: `20.000` deg
- candidate residual blend full alpha: `20.000` deg

## Regime Summary

- train/attached: count `2127`, Cm mean `0.07148`, Cm MAE `0.07777`, CY mean `-0.17464`, Cl mean `0.00912`, Cn mean `-0.00489`
- train/transition: count `6603`, Cm mean `0.03366`, Cm MAE `0.04561`, CY mean `-0.06424`, Cl mean `-0.00899`, Cn mean `0.00340`
- train/post_stall: count `916`, Cm mean `-0.00669`, Cm MAE `0.05599`, CY mean `-0.07054`, Cl mean `-0.02602`, Cn mean `0.00836`
- heldout/attached: count `367`, Cm mean `0.07180`, Cm MAE `0.07558`, CY mean `-0.22409`, Cl mean `0.01467`, Cn mean `-0.00240`
- heldout/transition: count `1300`, Cm mean `0.03119`, Cm MAE `0.04218`, CY mean `-0.07566`, Cl mean `-0.01320`, Cn mean `0.00566`
- heldout/post_stall: count `121`, Cm mean `-0.02355`, Cm MAE `0.04547`, CY mean `-0.03690`, Cl mean `-0.04273`, Cn mean `0.01100`

## Independent Stage Fit Summary

Transition is split into samples before and after the first post-stall exposure in each throw. This separates real transition-model evidence from transition samples that may already be contaminated by earlier post-stall divergence.

- train/attached: count `2127`, Cm mean `0.07148`, Cm MAE `0.07777`, Cm fit residual MAE `0.04889`, CY mean `-0.17464`, Cl mean `0.00912`, Cn mean `-0.00489`
- train/transition_before_post_stall: count `5758`, Cm mean `0.03271`, Cm MAE `0.04376`, Cm fit residual MAE `0.03886`, CY mean `-0.08358`, Cl mean `-0.00965`, Cn mean `0.00326`
- train/transition_after_post_stall: count `845`, Cm mean `0.04012`, Cm MAE `0.05820`, Cm fit residual MAE `0.06409`, CY mean `0.06761`, Cl mean `-0.00447`, Cn mean `0.00441`
- train/post_stall: count `916`, Cm mean `-0.00669`, Cm MAE `0.05599`, Cm fit residual MAE `0.03941`, CY mean `-0.07054`, Cl mean `-0.02602`, Cn mean `0.00836`
- heldout/attached: count `367`, Cm mean `0.07180`, Cm MAE `0.07558`, Cm fit residual MAE `0.04673`, CY mean `-0.22409`, Cl mean `0.01467`, Cn mean `-0.00240`
- heldout/transition_before_post_stall: count `1200`, Cm mean `0.03134`, Cm MAE `0.04177`, Cm fit residual MAE `0.03684`, CY mean `-0.09275`, Cl mean `-0.01313`, Cn mean `0.00556`
- heldout/transition_after_post_stall: count `100`, Cm mean `0.02934`, Cm MAE `0.04702`, Cm fit residual MAE `0.05574`, CY mean `0.12938`, Cl mean `-0.01400`, Cn mean `0.00688`
- heldout/post_stall: count `121`, Cm mean `-0.02355`, Cm MAE `0.04547`, Cm fit residual MAE `0.02309`, CY mean `-0.03690`, Cl mean `-0.04273`, Cn mean `0.01100`

## Interpretation

Accept the primary candidate only for the longitudinal claim-bearing model when held-out dx, altitude loss, sink, and pitch improve or preserve the active baseline. Treat dy, roll, and yaw as reported residual evidence unless the secondary lateral diagnostic improves held-out dy/roll/yaw without damaging those longitudinal metrics.
