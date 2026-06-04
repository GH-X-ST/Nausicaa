# Neutral Aero Residual Regime Fit

This run uses only neutral open-loop real throws. It estimates 6-DoF force/moment residuals from Vicon state trajectories, keeps attached and transition Cm offsets report-only, fits compact post-stall alpha-RBF residual surfaces for CL/CD/Cm/Cmq plus CY/Cl/Cn lateral-directional coupling, then validates the candidate by held-out dry-air replay.

## Rerun Recipe

- source session root: `04_Flight_Test\05_Results\cal\n30`
- alignment window: `0.020` s
- derivative window: `0.040` s
- replay dt: `0.0050` s
- ridge lambda: `0.001`
- min speed: `1.50` m/s
- workers: `8`
- apply attached Cm bias: `False`
- fit post-stall damping: `True`

```powershell
python 03_Control/02_Inner_Loop/run_fit_neutral_aero_residual_calibration.py --session-root 04_Flight_Test\05_Results\cal\n30 --run-label n30_full6dof_residual_align0p02_h10_v1 --heldout-count 10 --heldout-seed 606 --alignment-window-s 0.02 --derivative-window-s 0.04 --replay-dt-s 0.005 --ridge-lambda 0.001 --min-speed-m-s 1.5 --workers 8 --no-apply-attached-cm-bias --fit-post-stall-damping
```

## Coefficient Fit

- fit status: `ok`
- sample count: `151`
- used sample count: `151`
- post-stall used sample count: `114`
- fit MAE in Cm: `0.03024`
- attached Cm residual: `0.0306861`
- transition Cm residual before post-stall: `0.0368054`
- transition Cm residual after post-stall: `0`
- post-stall surface centres: `20, 45, 70` deg
- post-stall surface width: `15` deg
- post-stall CL surface: 20 deg `0.545137`, 45 deg `-1.1083`, 70 deg `1.18587`
- post-stall CD surface: 20 deg `0.201397`, 45 deg `-0.302685`, 70 deg `-1.5`
- post-stall Cm surface: 20 deg `0.152404`, 45 deg `-0.468109`, 70 deg `1.05191`
- post-stall Cmq surface: 20 deg `2.81058`, 45 deg `-0.66579`, 70 deg `-0.212917`
- post-stall CY surface: bias: 20 deg `-0.131056`, 45 deg `0.521784`, 70 deg `0.355545`; beta: 20 deg `-1.01746`, 45 deg `-2`, 70 deg `0.331608`; p_hat: 20 deg `1.57135`, 45 deg `-0.473748`, 70 deg `0.0290166`; r_hat: 20 deg `-2.95808`, 45 deg `1.75434`, 70 deg `0.272229`
- post-stall Cl surface: bias: 20 deg `0.0182316`, 45 deg `-0.018146`, 70 deg `0.085621`; beta: 20 deg `0.0129861`, 45 deg `-0.245164`, 70 deg `0.0550539`; p_hat: 20 deg `0.465808`, 45 deg `0.303656`, 70 deg `0.0541919`; r_hat: 20 deg `-0.456605`, 45 deg `0.0800239`, 70 deg `0.0121828`
- post-stall Cn surface: bias: 20 deg `-0.0342746`, 45 deg `0.0662549`, 70 deg `-0.0923517`; beta: 20 deg `-0.0618453`, 45 deg `0.424603`, 70 deg `-0.0215786`; p_hat: 20 deg `-0.152195`, 45 deg `0.299796`, 70 deg `0.0215978`; r_hat: 20 deg `0.256543`, 45 deg `-0.314643`, 70 deg `-0.0454232`
- post-stall Cmq residual: `0`
- selected post-stall surface replay scale: `0.750`
- transition blender status: `ok`
- transition blender fit group: `transition_before_post_stall`
- transition blender start alpha: `10.000` deg
- transition blender full alpha: `20.000` deg

## Replay Validation

- baseline train pitch MAE: `18.464` deg
- candidate train pitch MAE: `2.372` deg
- baseline held-out pitch MAE: `30.717` deg
- candidate held-out pitch MAE: `50.068` deg
- baseline held-out altitude-loss MAE: `0.7944` m
- candidate held-out altitude-loss MAE: `0.8055` m
- baseline held-out dx MAE: `1.3603` m
- candidate held-out dx MAE: `0.9469` m

## Stage Replay Errors

These are sample-aligned replay residuals grouped by the measured Vicon alpha regime. Use them to see whether the candidate fixes attached, transition, or post-stall behaviour; keep the full-throw held-out replay gate as the acceptance criterion.

- train/attached: samples `11`, throws `1`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.019` m, `0.002` m, `0.005` m, `0.143` m/s, `0.86` deg, `0.97` deg, `0.25` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.020` m, `0.002` m, `0.005` m, `0.142` m/s, `0.86` deg, `0.95` deg, `0.24` deg
- train/transition: samples `25`, throws `1`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.057` m, `0.034` m, `0.094` m, `0.395` m/s, `3.35` deg, `6.76` deg, `2.83` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.067` m, `0.009` m, `0.072` m, `0.310` m/s, `5.19` deg, `3.77` deg, `1.76` deg
- train/post_stall: samples `114`, throws `1`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.148` m, `0.274` m, `0.345` m, `0.517` m/s, `15.51` deg, `10.18` deg, `13.93` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.062` m, `0.022` m, `0.269` m, `0.388` m/s, `11.17` deg, `4.39` deg, `2.67` deg
- heldout/attached: no replay samples
- heldout/transition: samples `11`, throws `1`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.277` m, `0.008` m, `0.024` m, `0.667` m/s, `0.17` deg, `5.09` deg, `0.26` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.277` m, `0.009` m, `0.024` m, `0.666` m/s, `0.18` deg, `5.08` deg, `0.27` deg
- heldout/post_stall: samples `141`, throws `1`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `1.030` m, `0.038` m, `0.578` m, `1.088` m/s, `16.57` deg, `10.24` deg, `12.05` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.863` m, `0.063` m, `0.598` m, `1.128` m/s, `18.33` deg, `18.29` deg, `12.41` deg

## Candidate Parameters

- legacy scalar post-stall CL residual: baseline `0`, candidate `0`
- legacy scalar post-stall CD residual: baseline `0`, candidate `0`
- legacy scalar post-stall Cm residual: baseline `0`, candidate `0`
- post-stall CL surface: 20 deg baseline `0` -> candidate `0.408853`, 45 deg baseline `0` -> candidate `-0.831224`, 70 deg baseline `0` -> candidate `0.889402`
- post-stall CD surface: 20 deg baseline `0` -> candidate `0.151048`, 45 deg baseline `0` -> candidate `-0.227014`, 70 deg baseline `0` -> candidate `-1.125`
- post-stall Cm surface: 20 deg baseline `0` -> candidate `0.114303`, 45 deg baseline `0` -> candidate `-0.351082`, 70 deg baseline `0` -> candidate `0.788935`
- post-stall Cmq surface: 20 deg baseline `0` -> candidate `2.10793`, 45 deg baseline `0` -> candidate `-0.499343`, 70 deg baseline `0` -> candidate `-0.159688`
- post-stall CY surface: bias: 20 deg baseline `0` -> candidate `-0.0982918`, 45 deg baseline `0` -> candidate `0.391338`, 70 deg baseline `0` -> candidate `0.266659`; beta: 20 deg baseline `0` -> candidate `-0.763096`, 45 deg baseline `0` -> candidate `-1.5`, 70 deg baseline `0` -> candidate `0.248706`; p_hat: 20 deg baseline `0` -> candidate `1.17851`, 45 deg baseline `0` -> candidate `-0.355311`, 70 deg baseline `0` -> candidate `0.0217624`; r_hat: 20 deg baseline `0` -> candidate `-2.21856`, 45 deg baseline `0` -> candidate `1.31575`, 70 deg baseline `0` -> candidate `0.204172`
- post-stall Cl surface: bias: 20 deg baseline `0` -> candidate `0.0136737`, 45 deg baseline `0` -> candidate `-0.0136095`, 70 deg baseline `0` -> candidate `0.0642158`; beta: 20 deg baseline `0` -> candidate `0.0097396`, 45 deg baseline `0` -> candidate `-0.183873`, 70 deg baseline `0` -> candidate `0.0412904`; p_hat: 20 deg baseline `0` -> candidate `0.349356`, 45 deg baseline `0` -> candidate `0.227742`, 70 deg baseline `0` -> candidate `0.040644`; r_hat: 20 deg baseline `0` -> candidate `-0.342454`, 45 deg baseline `0` -> candidate `0.0600179`, 70 deg baseline `0` -> candidate `0.00913713`
- post-stall Cn surface: bias: 20 deg baseline `0` -> candidate `-0.025706`, 45 deg baseline `0` -> candidate `0.0496912`, 70 deg baseline `0` -> candidate `-0.0692638`; beta: 20 deg baseline `0` -> candidate `-0.046384`, 45 deg baseline `0` -> candidate `0.318452`, 70 deg baseline `0` -> candidate `-0.016184`; p_hat: 20 deg baseline `0` -> candidate `-0.114146`, 45 deg baseline `0` -> candidate `0.224847`, 70 deg baseline `0` -> candidate `0.0161983`; r_hat: 20 deg baseline `0` -> candidate `0.192407`, 45 deg baseline `0` -> candidate `-0.235982`, 70 deg baseline `0` -> candidate `-0.0340674`
- baseline post-stall Cmq: `0`
- candidate post-stall Cmq: `0`
- baseline residual blend start alpha: `12.000` deg
- candidate residual blend start alpha: `10.000` deg
- baseline residual blend full alpha: `20.000` deg
- candidate residual blend full alpha: `20.000` deg

## Regime Summary

- train/attached: count `12`, Cm mean `0.03061`, Cm MAE `0.03061`, CY mean `-0.12126`, Cl mean `0.01821`, Cn mean `-0.01058`
- train/transition: count `25`, Cm mean `0.03096`, Cm MAE `0.03373`, CY mean `-0.19888`, Cl mean `-0.00044`, Cn mean `-0.01074`
- train/post_stall: count `114`, Cm mean `-0.15747`, Cm MAE `0.17065`, CY mean `0.25031`, Cl mean `-0.01634`, Cn mean `-0.00725`
- heldout/attached: count `0`, Cm mean `nan`, Cm MAE `nan`, CY mean `nan`, Cl mean `nan`, Cn mean `nan`
- heldout/transition: count `12`, Cm mean `0.14926`, Cm MAE `0.14926`, CY mean `-0.47506`, Cl mean `-0.00702`, Cn mean `0.00223`
- heldout/post_stall: count `141`, Cm mean `-0.22200`, Cm MAE `0.24180`, CY mean `0.74003`, Cl mean `-0.00588`, Cn mean `-0.01133`

## Independent Stage Fit Summary

Transition is split into samples before and after the first post-stall exposure in each throw. This separates real transition-model evidence from transition samples that may already be contaminated by earlier post-stall divergence.

- train/attached: count `12`, Cm mean `0.03061`, Cm MAE `0.03061`, Cm fit residual MAE `0.01679`, CY mean `-0.12126`, Cl mean `0.01821`, Cn mean `-0.01058`
- train/transition_before_post_stall: count `25`, Cm mean `0.03096`, Cm MAE `0.03373`, Cm fit residual MAE `0.04536`, CY mean `-0.19888`, Cl mean `-0.00044`, Cn mean `-0.01074`
- train/transition_after_post_stall: count `0`, Cm mean `nan`, Cm MAE `nan`, Cm fit residual MAE `nan`, CY mean `nan`, Cl mean `nan`, Cn mean `nan`
- train/post_stall: count `114`, Cm mean `-0.15747`, Cm MAE `0.17065`, Cm fit residual MAE `0.02834`, CY mean `0.25031`, Cl mean `-0.01634`, Cn mean `-0.00725`
- heldout/attached: count `0`, Cm mean `nan`, Cm MAE `nan`, Cm fit residual MAE `nan`, CY mean `nan`, Cl mean `nan`, Cn mean `nan`
- heldout/transition_before_post_stall: count `12`, Cm mean `0.14926`, Cm MAE `0.14926`, Cm fit residual MAE `0.04875`, CY mean `-0.47506`, Cl mean `-0.00702`, Cn mean `0.00223`
- heldout/transition_after_post_stall: count `0`, Cm mean `nan`, Cm MAE `nan`, Cm fit residual MAE `nan`, CY mean `nan`, Cl mean `nan`, Cn mean `nan`
- heldout/post_stall: count `141`, Cm mean `-0.22200`, Cm MAE `0.24180`, Cm fit residual MAE `0.74116`, CY mean `0.74003`, Cl mean `-0.00588`, Cn mean `-0.01133`

## Interpretation

Accept the candidate only if held-out replay improves pitch/q and lateral trajectory behaviour without damaging x, altitude loss, or sink. Attached and transition Cm residuals are diagnostic-only by default; accepted high-AoA changes should enter through the compact alpha-dependent post-stall static, pitch-damping, and lateral-directional residual surfaces.
