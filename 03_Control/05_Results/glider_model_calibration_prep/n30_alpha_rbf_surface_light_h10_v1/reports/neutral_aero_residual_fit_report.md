# Neutral Aero Residual Regime Fit

This run uses only neutral open-loop real throws. It estimates force/moment residuals from Vicon state trajectories, keeps attached and transition Cm offsets report-only, fits a compact post-stall alpha-RBF residual surface for CL/CD/Cm, then validates the candidate by held-out dry-air replay.

## Rerun Recipe

- source session root: `04_Flight_Test\05_Results\cal\n30`
- alignment window: `0.100` s
- derivative window: `0.040` s
- replay dt: `0.0050` s
- ridge lambda: `0.001`
- min speed: `1.50` m/s
- workers: `8`
- apply attached Cm bias: `False`
- fit post-stall damping: `False`

```powershell
python 03_Control/02_Inner_Loop/run_fit_neutral_aero_residual_calibration.py --session-root 04_Flight_Test\05_Results\cal\n30 --run-label n30_alpha_rbf_surface_light_h10_v1 --heldout-count 10 --heldout-seed 606 --alignment-window-s 0.1 --derivative-window-s 0.04 --replay-dt-s 0.005 --ridge-lambda 0.001 --min-speed-m-s 1.5 --workers 8 --no-apply-attached-cm-bias --no-fit-post-stall-damping
```

## Coefficient Fit

- fit status: `ok`
- sample count: `3519`
- used sample count: `3519`
- post-stall used sample count: `2806`
- fit MAE in Cm: `0.07819`
- attached Cm residual: `0.0219436`
- transition Cm residual before post-stall: `0.0579849`
- transition Cm residual after post-stall: `-0.105133`
- post-stall surface centres: `20, 45, 70` deg
- post-stall surface width: `15` deg
- post-stall CL surface: 20 deg `0.358242`, 45 deg `-0.310715`, 70 deg `-0.1545`
- post-stall CD surface: 20 deg `0.112044`, 45 deg `0.0161129`, 70 deg `-0.818494`
- post-stall Cm surface: 20 deg `-0.00525029`, 45 deg `-0.136606`, 70 deg `-0.218807`
- post-stall Cmq residual: `0`
- transition blender status: `ok`
- transition blender fit group: `transition_before_post_stall`
- transition blender start alpha: `11.000` deg
- transition blender full alpha: `20.000` deg

## Replay Validation

- baseline train pitch MAE: `16.470` deg
- candidate train pitch MAE: `8.988` deg
- baseline held-out pitch MAE: `18.886` deg
- candidate held-out pitch MAE: `12.859` deg
- baseline held-out altitude-loss MAE: `0.3265` m
- candidate held-out altitude-loss MAE: `0.4467` m
- baseline held-out dx MAE: `0.7509` m
- candidate held-out dx MAE: `0.7250` m

## Stage Replay Errors

These are sample-aligned replay residuals grouped by the measured Vicon alpha regime. Use them to see whether the candidate fixes attached, transition, or post-stall behaviour; keep the full-throw held-out replay gate as the acceptance criterion.

- train/attached: samples `116`, throws `14`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.097` m, `0.009` m, `0.011` m, `0.297` m/s, `0.66` deg, `2.27` deg, `0.20` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.097` m, `0.009` m, `0.011` m, `0.295` m/s, `0.66` deg, `2.27` deg, `0.20` deg
- train/transition: samples `532`, throws `20`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.085` m, `0.100` m, `0.078` m, `0.388` m/s, `3.51` deg, `4.80` deg, `0.94` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.077` m, `0.108` m, `0.062` m, `0.327` m/s, `4.24` deg, `4.69` deg, `1.00` deg
- train/post_stall: samples `2851`, throws `20`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.350` m, `0.561` m, `0.273` m, `0.438` m/s, `23.30` deg, `10.20` deg, `9.47` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.286` m, `0.641` m, `0.202` m, `0.312` m/s, `27.62` deg, `11.38` deg, `9.89` deg
- heldout/attached: samples `65`, throws `8`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.019` m, `0.011` m, `0.010` m, `0.247` m/s, `0.54` deg, `1.63` deg, `0.31` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.019` m, `0.011` m, `0.010` m, `0.247` m/s, `0.54` deg, `1.63` deg, `0.31` deg
- heldout/transition: samples `265`, throws `10`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.209` m, `0.241` m, `0.184` m, `0.479` m/s, `7.09` deg, `9.90` deg, `3.75` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.203` m, `0.246` m, `0.170` m, `0.432` m/s, `7.49` deg, `9.94` deg, `3.89` deg
- heldout/post_stall: samples `1420`, throws `10`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.423` m, `0.766` m, `0.277` m, `0.412` m/s, `25.75` deg, `12.03` deg, `9.05` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.367` m, `0.838` m, `0.233` m, `0.325` m/s, `29.16` deg, `11.93` deg, `8.80` deg

## Candidate Parameters

- legacy scalar post-stall CL residual: baseline `0`, candidate `0`
- legacy scalar post-stall CD residual: baseline `0`, candidate `0`
- legacy scalar post-stall Cm residual: baseline `0`, candidate `0`
- post-stall CL surface: 20 deg baseline `0` -> candidate `0.358242`, 45 deg baseline `0` -> candidate `-0.310715`, 70 deg baseline `0` -> candidate `-0.1545`
- post-stall CD surface: 20 deg baseline `0` -> candidate `0.112044`, 45 deg baseline `0` -> candidate `0.0161129`, 70 deg baseline `0` -> candidate `-0.818494`
- post-stall Cm surface: 20 deg baseline `0` -> candidate `-0.00525029`, 45 deg baseline `0` -> candidate `-0.136606`, 70 deg baseline `0` -> candidate `-0.218807`
- baseline post-stall Cmq: `0`
- candidate post-stall Cmq: `0`
- baseline residual blend start alpha: `12.000` deg
- candidate residual blend start alpha: `11.000` deg
- baseline residual blend full alpha: `20.000` deg
- candidate residual blend full alpha: `20.000` deg

## Regime Summary

- train/attached: count `130`, Cm mean `0.02802`, Cm MAE `0.02929`
- train/transition: count `538`, Cm mean `0.05084`, Cm MAE `0.06494`
- train/post_stall: count `2851`, Cm mean `-0.15531`, Cm MAE `0.16575`
- heldout/attached: count `73`, Cm mean `0.03199`, Cm MAE `0.03342`
- heldout/transition: count `267`, Cm mean `0.00843`, Cm MAE `0.09653`
- heldout/post_stall: count `1420`, Cm mean `-0.14198`, Cm MAE `0.15595`

## Independent Stage Fit Summary

Transition is split into samples before and after the first post-stall exposure in each throw. This separates real transition-model evidence from transition samples that may already be contaminated by earlier post-stall divergence.

- train/attached: count `130`, Cm mean `0.02802`, Cm MAE `0.02929`, Cm fit residual MAE `0.01854`
- train/transition_before_post_stall: count `510`, Cm mean `0.05939`, Cm MAE `0.06275`, Cm fit residual MAE `0.07865`
- train/transition_after_post_stall: count `28`, Cm mean `-0.10479`, Cm MAE `0.10479`, Cm fit residual MAE `0.07264`
- train/post_stall: count `2851`, Cm mean `-0.15531`, Cm MAE `0.16575`, Cm fit residual MAE `0.08089`
- heldout/attached: count `73`, Cm mean `0.03199`, Cm MAE `0.03342`, Cm fit residual MAE `0.02573`
- heldout/transition_before_post_stall: count `210`, Cm mean `0.06454`, Cm MAE `0.06891`, Cm fit residual MAE `0.08524`
- heldout/transition_after_post_stall: count `57`, Cm mean `-0.19830`, Cm MAE `0.19830`, Cm fit residual MAE `0.16703`
- heldout/post_stall: count `1420`, Cm mean `-0.14198`, Cm MAE `0.15595`, Cm fit residual MAE `0.07032`

## Interpretation

Accept the candidate only if held-out replay improves pitch/q behaviour without damaging x, altitude loss, or sink. Attached and transition Cm residuals are diagnostic-only by default; accepted high-AoA changes should enter through the compact alpha-dependent post-stall coefficient surface.
