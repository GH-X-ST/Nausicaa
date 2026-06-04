# Neutral Aero Residual Regime Fit

This run uses only neutral open-loop real throws. It estimates 6-DoF force/moment residuals from Vicon state trajectories, keeps attached and transition Cm offsets report-only, fits compact post-stall alpha-RBF residual surfaces for CL/CD/Cm/Cmq plus CY/Cl/Cn lateral-directional coupling, then validates the candidate by held-out dry-air replay.

## Rerun Recipe

- source session root: `04_Flight_Test\05_Results\cal\n30`
- alignment window: `0.100` s
- derivative window: `0.040` s
- replay dt: `0.0050` s
- ridge lambda: `0.001`
- min speed: `1.50` m/s
- workers: `8`
- aligned launch filter: `True`
- aligned launch filter bounds: `u=[4.00, 8.00]` m/s, `|v|<=1.50` m/s, `|w|<=0.90` m/s
- apply attached Cm bias: `True`
- fit post-stall damping: `True`
- fit lateral surfaces: `False`

```powershell
python 03_Control/02_Inner_Loop/run_fit_neutral_aero_residual_calibration.py --session-root 04_Flight_Test\05_Results\cal\n30 --run-label n30_b105_cm_only_filt_h10 --heldout-count 10 --heldout-seed 606 --alignment-window-s 0.1 --derivative-window-s 0.04 --replay-dt-s 0.005 --ridge-lambda 0.001 --min-speed-m-s 1.5 --workers 8 --aligned-u-min-m-s 4 --aligned-u-max-m-s 8 --aligned-v-abs-max-m-s 1.5 --aligned-w-abs-max-m-s 0.9 --filter-aligned-launch-state --apply-attached-cm-bias --fit-post-stall-damping --no-fit-lateral-surfaces
```

## Aligned Launch Filter

- loaded logged-valid throws: `30`
- kept throws after replay-start filter: `20`
- filtered throws: `10`
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

## Coefficient Fit

- fit status: `ok`
- sample count: `1711`
- used sample count: `1711`
- post-stall used sample count: `0`
- fit MAE in Cm: `0.04860`
- attached Cm residual: `0.0935528`
- transition Cm residual before post-stall: `0.0565023`
- transition Cm residual after post-stall: `0.0284656`
- post-stall surface centres: `20, 45, 70` deg
- post-stall surface width: `15` deg
- post-stall CL surface: 20 deg `0.291583`, 45 deg `-1.20579`, 70 deg `1.5`
- post-stall CD surface: 20 deg `0.0723528`, 45 deg `-0.089279`, 70 deg `1.42177`
- post-stall Cm surface: 20 deg `-0.0210521`, 45 deg `0.294612`, 70 deg `-1.5`
- post-stall Cmq surface: 20 deg `1.80308`, 45 deg `3.57866`, 70 deg `0.204544`
- post-stall CY surface: bias: 20 deg `0`, 45 deg `0`, 70 deg `0`; beta: 20 deg `0`, 45 deg `0`, 70 deg `0`; p_hat: 20 deg `0`, 45 deg `0`, 70 deg `0`; r_hat: 20 deg `0`, 45 deg `0`, 70 deg `0`
- post-stall Cl surface: bias: 20 deg `0`, 45 deg `0`, 70 deg `0`; beta: 20 deg `0`, 45 deg `0`, 70 deg `0`; p_hat: 20 deg `0`, 45 deg `0`, 70 deg `0`; r_hat: 20 deg `0`, 45 deg `0`, 70 deg `0`
- post-stall Cn surface: bias: 20 deg `0`, 45 deg `0`, 70 deg `0`; beta: 20 deg `0`, 45 deg `0`, 70 deg `0`; p_hat: 20 deg `0`, 45 deg `0`, 70 deg `0`; r_hat: 20 deg `0`, 45 deg `0`, 70 deg `0`
- post-stall Cmq residual: `0`
- selected post-stall surface replay scale: `0.000`
- transition blender status: `ok`
- transition blender fit group: `transition_before_post_stall`
- transition blender start alpha: `12.000` deg
- transition blender full alpha: `19.000` deg

## Replay Validation

- baseline train pitch MAE: `20.191` deg
- candidate train pitch MAE: `16.401` deg
- baseline held-out pitch MAE: `15.644` deg
- candidate held-out pitch MAE: `15.518` deg
- baseline held-out altitude-loss MAE: `0.9277` m
- candidate held-out altitude-loss MAE: `0.0929` m
- baseline held-out dx MAE: `0.8684` m
- candidate held-out dx MAE: `0.3739` m

## Stage Replay Errors

These are sample-aligned replay residuals grouped by the measured Vicon alpha regime. Use them to see whether the candidate fixes attached, transition, or post-stall behaviour; keep the full-throw held-out replay gate as the acceptance criterion.

- train/attached: samples `317`, throws `10`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.171` m, `0.236` m, `0.196` m, `0.330` m/s, `4.46` deg, `7.68` deg, `3.08` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.058` m, `0.280` m, `0.030` m, `0.152` m/s, `7.30` deg, `6.48` deg, `6.14` deg
- train/transition: samples `1045`, throws `10`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.364` m, `0.528` m, `0.556` m, `0.786` m/s, `5.49` deg, `22.27` deg, `6.66` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.142` m, `0.609` m, `0.172` m, `0.309` m/s, `10.20` deg, `9.24` deg, `11.11` deg
- train/post_stall: samples `339`, throws `6`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.739` m, `0.995` m, `0.763` m, `0.923` m/s, `4.31` deg, `25.96` deg, `8.10` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.196` m, `1.210` m, `0.136` m, `0.198` m/s, `14.23` deg, `9.66` deg, `12.94` deg
- heldout/attached: samples `257`, throws `10`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.241` m, `0.421` m, `0.263` m, `0.392` m/s, `5.41` deg, `7.19` deg, `2.80` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.090` m, `0.501` m, `0.040` m, `0.181` m/s, `8.97` deg, `4.13` deg, `10.40` deg
- heldout/transition: samples `1234`, throws `10`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.363` m, `0.636` m, `0.404` m, `0.558` m/s, `12.57` deg, `15.86` deg, `8.90` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.089` m, `0.722` m, `0.116` m, `0.208` m/s, `19.52` deg, `9.18` deg, `18.90` deg
- heldout/post_stall: samples `172`, throws `4`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.420` m, `0.782` m, `0.450` m, `0.653` m/s, `15.39` deg, `19.48` deg, `5.35` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.054` m, `0.903` m, `0.141` m, `0.232` m/s, `22.43` deg, `6.03` deg, `14.49` deg

## Candidate Parameters

- legacy scalar post-stall CL residual: baseline `0`, candidate `0`
- legacy scalar post-stall CD residual: baseline `0`, candidate `0`
- legacy scalar post-stall Cm residual: baseline `0`, candidate `0`
- post-stall CL surface: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`
- post-stall CD surface: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`
- post-stall Cm surface: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`
- post-stall Cmq surface: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`
- post-stall CY surface: bias: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`; beta: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`; p_hat: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`; r_hat: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`
- post-stall Cl surface: bias: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`; beta: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`; p_hat: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`; r_hat: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`
- post-stall Cn surface: bias: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`; beta: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`; p_hat: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`; r_hat: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`
- baseline post-stall Cmq: `0`
- candidate post-stall Cmq: `0`
- baseline residual blend start alpha: `12.000` deg
- candidate residual blend start alpha: `12.000` deg
- baseline residual blend full alpha: `20.000` deg
- candidate residual blend full alpha: `19.000` deg

## Regime Summary

- train/attached: count `327`, Cm mean `0.08568`, Cm MAE `0.09418`, CY mean `-0.25535`, Cl mean `0.01372`, Cn mean `-0.00500`
- train/transition: count `1045`, Cm mean `0.03783`, Cm MAE `0.04882`, CY mean `-0.07880`, Cl mean `-0.01269`, Cn mean `0.00575`
- train/post_stall: count `339`, Cm mean `0.00226`, Cm MAE `0.05582`, CY mean `-0.04896`, Cl mean `-0.01817`, Cn mean `0.01056`
- heldout/attached: count `267`, Cm mean `0.04699`, Cm MAE `0.06289`, CY mean `-0.25190`, Cl mean `0.01089`, Cn mean `-0.00292`
- heldout/transition: count `1234`, Cm mean `0.03061`, Cm MAE `0.04758`, CY mean `-0.01179`, Cl mean `-0.00921`, Cn mean `0.00251`
- heldout/post_stall: count `172`, Cm mean `-0.00518`, Cm MAE `0.04452`, CY mean `-0.08629`, Cl mean `-0.04795`, Cn mean `0.01095`

## Independent Stage Fit Summary

Transition is split into samples before and after the first post-stall exposure in each throw. This separates real transition-model evidence from transition samples that may already be contaminated by earlier post-stall divergence.

- train/attached: count `327`, Cm mean `0.08568`, Cm MAE `0.09418`, Cm fit residual MAE `0.05952`, CY mean `-0.25535`, Cl mean `0.01372`, Cn mean `-0.00500`
- train/transition_before_post_stall: count `817`, Cm mean `0.04167`, Cm MAE `0.04691`, Cm fit residual MAE `0.04304`, CY mean `-0.13669`, Cl mean `-0.01505`, Cn mean `0.00721`
- train/transition_after_post_stall: count `228`, Cm mean `0.02406`, Cm MAE `0.05569`, Cm fit residual MAE `0.06014`, CY mean `0.12861`, Cl mean `-0.00424`, Cn mean `0.00049`
- train/post_stall: count `339`, Cm mean `0.00226`, Cm MAE `0.05582`, Cm fit residual MAE `0.04369`, CY mean `-0.04896`, Cl mean `-0.01817`, Cn mean `0.01056`
- heldout/attached: count `267`, Cm mean `0.04699`, Cm MAE `0.06289`, Cm fit residual MAE `0.06131`, CY mean `-0.25190`, Cl mean `0.01089`, Cn mean `-0.00292`
- heldout/transition_before_post_stall: count `1050`, Cm mean `0.03072`, Cm MAE `0.04719`, Cm fit residual MAE `0.04877`, CY mean `-0.03137`, Cl mean `-0.00928`, Cn mean `0.00173`
- heldout/transition_after_post_stall: count `184`, Cm mean `0.03002`, Cm MAE `0.04977`, Cm fit residual MAE `0.05925`, CY mean `0.09993`, Cl mean `-0.00882`, Cn mean `0.00697`
- heldout/post_stall: count `172`, Cm mean `-0.00518`, Cm MAE `0.04452`, Cm fit residual MAE `0.02414`, CY mean `-0.08629`, Cl mean `-0.04795`, Cn mean `0.01095`

## Interpretation

Accept the candidate only if held-out replay improves pitch/q and lateral trajectory behaviour without damaging x, altitude loss, or sink. Attached and transition Cm residuals are diagnostic-only by default; accepted high-AoA changes should enter through the compact alpha-dependent post-stall static, pitch-damping, and lateral-directional residual surfaces.
