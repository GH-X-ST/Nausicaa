# Neutral Aero Residual Regime Fit

This run uses only neutral open-loop real throws. It estimates 6-DoF force/moment residuals from Vicon state trajectories, fits first-class attached and transition-window CY/Cl/Cn coupling in beta, p_hat, and r_hat, fits compact post-stall alpha-RBF residual surfaces for CL/CD/Cm/Cmq plus separated-flow CY/Cl/Cn coupling, then cross-adjusts coefficient groups through train replay before validating the candidate by held-out dry-air replay.

## Rerun Recipe

- source session root: `04_Flight_Test\05_Results\cal\n30`
- alignment window: `0.100` s
- derivative window: `0.040` s
- replay dt: `0.0050` s
- ridge lambda: `0.001`
- min speed: `1.50` m/s
- workers: `8`
- fit workflow: `grouped_iterative`
- group iterations: `3`
- group improvement tolerance: `0.001`
- aligned launch filter: `True`
- aligned launch filter bounds: `u=[4.00, 8.00]` m/s, `|v|<=1.50` m/s, `|w|<=0.90` m/s
- apply attached Cm bias: `False`
- fit post-stall damping: `True`
- fit attached lateral coupling: `True`
- fit transition lateral coupling: `True`
- fit lateral surfaces: `True`

```powershell
python 03_Control/02_Inner_Loop/run_fit_neutral_aero_residual_calibration.py --session-root 04_Flight_Test\05_Results\cal\n30 --run-label n30_grouped_lateral_coupling_default_v1 --heldout-count 10 --heldout-seed 606 --alignment-window-s 0.1 --derivative-window-s 0.04 --replay-dt-s 0.005 --ridge-lambda 0.001 --min-speed-m-s 1.5 --workers 8 --fit-workflow grouped_iterative --group-iterations 3 --group-improvement-tol 0.001 --aligned-u-min-m-s 4 --aligned-u-max-m-s 8 --aligned-v-abs-max-m-s 1.5 --aligned-w-abs-max-m-s 0.9 --filter-aligned-launch-state --no-apply-attached-cm-bias --fit-post-stall-damping --fit-attached-lateral-coupling --fit-transition-lateral-coupling --fit-lateral-surfaces
```

## Aligned Launch Filter

- loaded logged-valid throws: `40`
- kept throws after replay-start filter: `27`
- filtered throws: `13`
- filter audit CSV: `metrics/neutral_aero_residual_filtered_throws.csv`
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

## Coefficient Fit

- fit status: `ok`
- sample count: `2848`
- used sample count: `2848`
- post-stall used sample count: `370`
- fit MAE in Cm: `0.04691`
- attached Cm residual: `0.0852854`
- transition Cm residual before post-stall: `0.0498098`
- transition Cm residual after post-stall: `0.0339979`
- attached lateral coupling:
  - CY: bias `-0.179935`, beta `-0.685718`, p_hat `-0.929104`, r_hat `1.56863`
  - Cl: bias `0.00272648`, beta `0.0252735`, p_hat `0.218107`, r_hat `-0.334078`
  - Cn: bias `0.00257702`, beta `-0.0281197`, p_hat `-0.0234002`, r_hat `0.25973`
- transition-window lateral coupling:
  - CY: bias `0.0579912`, beta `0.178988`, p_hat `0.834465`, r_hat `-3`
  - Cl: bias `-0.00459685`, beta `-0.0226917`, p_hat `-0.0935025`, r_hat `0.0687156`
  - Cn: bias `0.00591556`, beta `0.0190114`, p_hat `0.123488`, r_hat `-0.125952`
- post-stall surface centres: `20, 45, 70` deg
- post-stall surface width: `15` deg
- post-stall CL surface: 20 deg `0.204043`, 45 deg `-0.845126`, 70 deg `1.5`
- post-stall CD surface: 20 deg `0.0570136`, 45 deg `-0.0350041`, 70 deg `0.734928`
- post-stall Cm surface: 20 deg `0.0140023`, 45 deg `0.192334`, 70 deg `-1.5`
- post-stall Cmq surface: 20 deg `2.60096`, 45 deg `2.24781`, 70 deg `0.118929`
- post-stall CY surface: bias: 20 deg `0.446477`, 45 deg `-0.481824`, 70 deg `-1.86598`; beta: 20 deg `1.12784`, 45 deg `-1.00124`, 70 deg `1.4769`; p_hat: 20 deg `0.885198`, 45 deg `-0.557634`, 70 deg `-0.291916`; r_hat: 20 deg `-4`, 45 deg `-2.72654`, 70 deg `-0.0708634`
- post-stall Cl surface: bias: 20 deg `0.00450204`, 45 deg `-0.00507008`, 70 deg `-0.0938363`; beta: 20 deg `0.0406723`, 45 deg `-0.0509301`, 70 deg `0.047673`; p_hat: 20 deg `0.163721`, 45 deg `0.0538239`, 70 deg `-0.00282607`; r_hat: 20 deg `0.263495`, 45 deg `-0.155858`, 70 deg `-0.0129882`
- post-stall Cn surface: bias: 20 deg `-0.0105576`, 45 deg `0.0129451`, 70 deg `0.0400507`; beta: 20 deg `-0.0272742`, 45 deg `0.0735927`, 70 deg `-0.0300011`; p_hat: 20 deg `0.0769772`, 45 deg `0.127437`, 70 deg `0.0154953`; r_hat: 20 deg `-0.0537685`, 45 deg `0.0327869`, 70 deg `-0.00109049`
- post-stall Cmq residual: `0`
- selected post-stall surface replay scale: `0.500`
- transition blender status: `ok`
- transition blender fit group: `transition_before_post_stall`
- transition blender start alpha: `12.000` deg
- transition blender full alpha: `19.000` deg

## Grouped Replay Refinement

- grouped history rows: `61`
- grouped history CSV: `metrics/neutral_aero_residual_group_iteration_history.csv`
- selected grouped replay scales:
  - pass `0`, group `initial_grouped_candidate`, scale ``, objective `81.5720`
  - pass `1`, group `attached_lateral`, scale `1.25`, objective `38.6503`
  - pass `1`, group `post_stall_longitudinal`, scale `0.5`, objective `38.6503`
  - pass `1`, group `post_stall_lateral`, scale `0.75`, objective `38.5130`
  - pass `1`, group `transition_lateral`, scale `1.25`, objective `38.4783`
  - pass `1`, group `transition_blender`, scale `0`, objective `38.4316`
  - pass `2`, group `attached_lateral`, scale `1.25`, objective `38.4316`
  - pass `2`, group `post_stall_longitudinal`, scale `0.5`, objective `38.4316`
  - pass `2`, group `post_stall_lateral`, scale `0.75`, objective `38.4316`
  - pass `2`, group `transition_lateral`, scale `1.25`, objective `38.4316`
  - pass `2`, group `transition_blender`, scale `0`, objective `38.4316`

## Replay Validation

- baseline train pitch MAE: `26.369` deg
- candidate train pitch MAE: `48.503` deg
- baseline held-out pitch MAE: `14.709` deg
- candidate held-out pitch MAE: `50.659` deg
- baseline held-out altitude-loss MAE: `0.8549` m
- candidate held-out altitude-loss MAE: `0.2015` m
- baseline held-out dx MAE: `0.8293` m
- candidate held-out dx MAE: `0.3436` m
- held-out acceptance: `rejected_diagnostic_only`
- acceptance policy: held-out candidate must improve or preserve dx, dy, altitude loss, sink, roll, pitch, and yaw MAE versus baseline_active
  - dx_mae_m: baseline `0.8293`, candidate `0.3436`, delta `-0.4857`, pass `True`
  - dy_mae_m: baseline `1.4516`, candidate `1.0665`, delta `-0.3851`, pass `True`
  - altitude_loss_mae_m: baseline `0.8549`, candidate `0.2015`, delta `-0.6534`, pass `True`
  - sink_mae_m_s: baseline `0.7546`, candidate `0.1824`, delta `-0.5721`, pass `True`
  - final_phi_mae_deg: baseline `17.4110`, candidate `67.6913`, delta `50.2802`, pass `False`
  - final_theta_mae_deg: baseline `14.7090`, candidate `50.6586`, delta `35.9496`, pass `False`
  - final_psi_mae_deg: baseline `16.3238`, candidate `61.4668`, delta `45.1429`, pass `False`

## Stage Replay Errors

These are sample-aligned replay residuals grouped by the measured Vicon alpha regime. Use them to see whether the candidate fixes attached, transition, or post-stall behaviour; keep the full-throw held-out replay gate as the acceptance criterion.

- train/attached: samples `598`, throws `17`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.305` m, `0.452` m, `0.493` m, `0.588` m/s, `6.63` deg, `11.67` deg, `3.40` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.121` m, `0.337` m, `0.206` m, `0.294` m/s, `18.68` deg, `18.27` deg, `21.10` deg
- train/transition: samples `1855`, throws `17`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.380` m, `0.602` m, `0.574` m, `0.822` m/s, `6.27` deg, `23.31` deg, `6.80` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.200` m, `0.376` m, `0.363` m, `0.557` m/s, `24.53` deg, `26.39` deg, `32.63` deg
- train/post_stall: samples `377`, throws `8`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.682` m, `0.973` m, `0.740` m, `0.916` m/s, `4.17` deg, `26.20` deg, `8.03` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.173` m, `0.795` m, `0.332` m, `0.477` m/s, `35.23` deg, `33.29` deg, `55.48` deg
- heldout/attached: samples `296`, throws `10`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.242` m, `0.417` m, `0.294` m, `0.408` m/s, `5.58` deg, `8.02` deg, `2.63` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.095` m, `0.322` m, `0.063` m, `0.166` m/s, `13.57` deg, `15.39` deg, `14.12` deg
- heldout/transition: samples `1203`, throws `10`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.336` m, `0.630` m, `0.360` m, `0.515` m/s, `12.20` deg, `14.99` deg, `9.37` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.169` m, `0.351` m, `0.211` m, `0.330` m/s, `27.47` deg, `26.41` deg, `37.37` deg
- heldout/post_stall: samples `231`, throws `4`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.430` m, `0.761` m, `0.381` m, `0.565` m/s, `13.91` deg, `17.92` deg, `5.43` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.196` m, `0.391` m, `0.220` m, `0.320` m/s, `40.80` deg, `21.99` deg, `46.26` deg

## Candidate Parameters

- legacy scalar post-stall CL residual: baseline `0`, candidate `0`
- legacy scalar post-stall CD residual: baseline `0`, candidate `0`
- legacy scalar post-stall Cm residual: baseline `0`, candidate `0`
- post-stall CL surface: 20 deg baseline `0` -> candidate `0.102021`, 45 deg baseline `0` -> candidate `-0.422563`, 70 deg baseline `0` -> candidate `0.75`
- post-stall CD surface: 20 deg baseline `0` -> candidate `0.0285068`, 45 deg baseline `0` -> candidate `-0.0175021`, 70 deg baseline `0` -> candidate `0.367464`
- post-stall Cm surface: 20 deg baseline `0` -> candidate `0.00700113`, 45 deg baseline `0` -> candidate `0.0961669`, 70 deg baseline `0` -> candidate `-0.75`
- post-stall Cmq surface: 20 deg baseline `0` -> candidate `1.30048`, 45 deg baseline `0` -> candidate `1.1239`, 70 deg baseline `0` -> candidate `0.0594643`
- post-stall CY surface: bias: 20 deg baseline `0` -> candidate `0.334858`, 45 deg baseline `0` -> candidate `-0.361368`, 70 deg baseline `0` -> candidate `-1.39949`; beta: 20 deg baseline `0` -> candidate `0.845882`, 45 deg baseline `0` -> candidate `-0.750928`, 70 deg baseline `0` -> candidate `1.10767`; p_hat: 20 deg baseline `0` -> candidate `0.663898`, 45 deg baseline `0` -> candidate `-0.418226`, 70 deg baseline `0` -> candidate `-0.218937`; r_hat: 20 deg baseline `0` -> candidate `-3`, 45 deg baseline `0` -> candidate `-2.0449`, 70 deg baseline `0` -> candidate `-0.0531475`
- post-stall Cl surface: bias: 20 deg baseline `0` -> candidate `0.00337653`, 45 deg baseline `0` -> candidate `-0.00380256`, 70 deg baseline `0` -> candidate `-0.0703773`; beta: 20 deg baseline `0` -> candidate `0.0305042`, 45 deg baseline `0` -> candidate `-0.0381976`, 70 deg baseline `0` -> candidate `0.0357548`; p_hat: 20 deg baseline `0` -> candidate `0.122791`, 45 deg baseline `0` -> candidate `0.0403679`, 70 deg baseline `0` -> candidate `-0.00211956`; r_hat: 20 deg baseline `0` -> candidate `0.197622`, 45 deg baseline `0` -> candidate `-0.116893`, 70 deg baseline `0` -> candidate `-0.00974116`
- post-stall Cn surface: bias: 20 deg baseline `0` -> candidate `-0.00791818`, 45 deg baseline `0` -> candidate `0.00970879`, 70 deg baseline `0` -> candidate `0.030038`; beta: 20 deg baseline `0` -> candidate `-0.0204556`, 45 deg baseline `0` -> candidate `0.0551945`, 70 deg baseline `0` -> candidate `-0.0225008`; p_hat: 20 deg baseline `0` -> candidate `0.0577329`, 45 deg baseline `0` -> candidate `0.0955777`, 70 deg baseline `0` -> candidate `0.0116215`; r_hat: 20 deg baseline `0` -> candidate `-0.0403264`, 45 deg baseline `0` -> candidate `0.0245902`, 70 deg baseline `0` -> candidate `-0.000817868`
- attached lateral coupling: side_force_bias_coeff `0` -> `-0.224918`; side_force_beta_coeff `0` -> `-0.857148`; side_force_p_hat_coeff `0` -> `-1.16138`; side_force_r_hat_coeff `0` -> `1.96079`; roll_moment_bias_coeff `0` -> `0.0034081`; roll_moment_beta_coeff `0` -> `0.0315919`; roll_moment_p_hat_coeff `0` -> `0.272634`; roll_moment_r_hat_coeff `0` -> `-0.417597`; yaw_moment_bias_coeff `0` -> `0.00322127`; yaw_moment_beta_coeff `0` -> `-0.0351496`; yaw_moment_p_hat_coeff `0` -> `-0.0292503`; yaw_moment_r_hat_coeff `0` -> `0.324663`
- transition lateral coupling: transition_side_force_bias_coeff `0` -> `0.072489`; transition_side_force_beta_coeff `0` -> `0.223734`; transition_side_force_p_hat_coeff `0` -> `1.04308`; transition_side_force_r_hat_coeff `0` -> `-3`; transition_roll_moment_bias_coeff `0` -> `-0.00574606`; transition_roll_moment_beta_coeff `0` -> `-0.0283646`; transition_roll_moment_p_hat_coeff `0` -> `-0.116878`; transition_roll_moment_r_hat_coeff `0` -> `0.0858945`; transition_yaw_moment_bias_coeff `0` -> `0.00739445`; transition_yaw_moment_beta_coeff `0` -> `0.0237642`; transition_yaw_moment_p_hat_coeff `0` -> `0.154361`; transition_yaw_moment_r_hat_coeff `0` -> `-0.15744`
- baseline post-stall Cmq: `0`
- candidate post-stall Cmq: `0`
- baseline residual blend start alpha: `12.000` deg
- candidate residual blend start alpha: `12.000` deg
- baseline residual blend full alpha: `20.000` deg
- candidate residual blend full alpha: `20.000` deg

## Regime Summary

- train/attached: count `616`, Cm mean `0.07296`, Cm MAE `0.08370`, CY mean `-0.21183`, Cl mean `0.01172`, Cn mean `-0.00294`
- train/transition: count `1855`, Cm mean `0.03672`, Cm MAE `0.04883`, CY mean `-0.07636`, Cl mean `-0.01191`, Cn mean `0.00718`
- train/post_stall: count `377`, Cm mean `0.00513`, Cm MAE `0.05491`, CY mean `-0.06650`, Cl mean `-0.02050`, Cn mean `0.01134`
- heldout/attached: count `306`, Cm mean `0.05244`, Cm MAE `0.06678`, CY mean `-0.25167`, Cl mean `0.01241`, Cn mean `-0.00317`
- heldout/transition: count `1203`, Cm mean `0.03251`, Cm MAE `0.04719`, CY mean `-0.04437`, Cl mean `-0.01008`, Cn mean `0.00353`
- heldout/post_stall: count `231`, Cm mean `-0.01805`, Cm MAE `0.04800`, CY mean `-0.01579`, Cl mean `-0.04343`, Cn mean `0.00627`

## Independent Stage Fit Summary

Transition is split into samples before and after the first post-stall exposure in each throw. This separates real transition-model evidence from transition samples that may already be contaminated by earlier post-stall divergence.

- train/attached: count `616`, Cm mean `0.07296`, Cm MAE `0.08370`, Cm fit residual MAE `0.05812`, CY mean `-0.21183`, Cl mean `0.01172`, Cn mean `-0.00294`
- train/transition_before_post_stall: count `1506`, Cm mean `0.03838`, Cm MAE `0.04757`, Cm fit residual MAE `0.04142`, CY mean `-0.11405`, Cl mean `-0.01324`, Cn mean `0.00777`
- train/transition_after_post_stall: count `349`, Cm mean `0.02956`, Cm MAE `0.05424`, Cm fit residual MAE `0.06007`, CY mean `0.08628`, Cl mean `-0.00616`, Cn mean `0.00462`
- train/post_stall: count `377`, Cm mean `0.00513`, Cm MAE `0.05491`, Cm fit residual MAE `0.03834`, CY mean `-0.06650`, Cl mean `-0.02050`, Cn mean `0.01134`
- heldout/attached: count `306`, Cm mean `0.05244`, Cm MAE `0.06678`, Cm fit residual MAE `0.05612`, CY mean `-0.25167`, Cl mean `0.01241`, Cn mean `-0.00317`
- heldout/transition_before_post_stall: count `1028`, Cm mean `0.03136`, Cm MAE `0.04584`, Cm fit residual MAE `0.04590`, CY mean `-0.06594`, Cl mean `-0.01124`, Cn mean `0.00188`
- heldout/transition_after_post_stall: count `175`, Cm mean `0.03922`, Cm MAE `0.05510`, Cm fit residual MAE `0.06331`, CY mean `0.08228`, Cl mean `-0.00327`, Cn mean `0.01323`
- heldout/post_stall: count `231`, Cm mean `-0.01805`, Cm MAE `0.04800`, Cm fit residual MAE `0.02651`, CY mean `-0.01579`, Cl mean `-0.04343`, Cn mean `0.00627`

## Interpretation

Accept the candidate only if held-out replay improves or preserves dx, dy, altitude loss, sink, roll, pitch, and yaw. Attached Cm remains diagnostic-only by default; lateral coupling is accepted only when the full held-out gate passes.
