# Neutral Aero Residual Regime Fit

This run uses only neutral open-loop real throws. It estimates 6-DoF force/moment residuals from Vicon state trajectories, keeps attached and transition Cm offsets report-only, fits compact post-stall alpha-RBF residual surfaces for CL/CD/Cm/Cmq plus CY/Cl/Cn lateral-directional coupling, then validates the candidate by held-out dry-air replay.

## Rerun Recipe

- source session root: `04_Flight_Test\05_Results\cal\n30`
- alignment window: `0.100` s
- derivative window: `0.040` s
- replay dt: `0.0050` s
- ridge lambda: `0.01`
- min speed: `1.50` m/s
- workers: `8`
- apply attached Cm bias: `False`
- fit post-stall damping: `True`

```powershell
python 03_Control/02_Inner_Loop/run_fit_neutral_aero_residual_calibration.py --session-root 04_Flight_Test\05_Results\cal\n30 --run-label n30_full6dof_ridge1e2_h10_v1 --heldout-count 10 --heldout-seed 606 --alignment-window-s 0.1 --derivative-window-s 0.04 --replay-dt-s 0.005 --ridge-lambda 0.01 --min-speed-m-s 1.5 --workers 8 --no-apply-attached-cm-bias --fit-post-stall-damping
```

## Coefficient Fit

- fit status: `ok`
- sample count: `3519`
- used sample count: `3519`
- post-stall used sample count: `2805`
- fit MAE in Cm: `0.06240`
- attached Cm residual: `0.0191477`
- transition Cm residual before post-stall: `0.0588538`
- transition Cm residual after post-stall: `-0.1051`
- post-stall surface centres: `20, 45, 70` deg
- post-stall surface width: `15` deg
- post-stall CL surface: 20 deg `0.362678`, 45 deg `-0.339429`, 70 deg `-0.139638`
- post-stall CD surface: 20 deg `0.114787`, 45 deg `0.000251037`, 70 deg `-0.808125`
- post-stall Cm surface: 20 deg `-0.0304711`, 45 deg `-0.0784299`, 70 deg `-0.0321721`
- post-stall Cmq surface: 20 deg `4`, 45 deg `-0.728968`, 70 deg `4`
- post-stall CY surface: bias: 20 deg `-0.387403`, 45 deg `0.416886`, 70 deg `0.282308`; beta: 20 deg `-0.733037`, 45 deg `-0.501251`, 70 deg `-1.33274`; p_hat: 20 deg `-2.11186`, 45 deg `-0.68365`, 70 deg `-0.6132`; r_hat: 20 deg `-4`, 45 deg `-1.14904`, 70 deg `-4`
- post-stall Cl surface: bias: 20 deg `-0.00864305`, 45 deg `0.00471959`, 70 deg `0.0349746`; beta: 20 deg `0.00836508`, 45 deg `-0.0687471`, 70 deg `0.0837384`; p_hat: 20 deg `0.23809`, 45 deg `0.403604`, 70 deg `0.30329`; r_hat: 20 deg `-0.461466`, 45 deg `0.0729704`, 70 deg `-0.784011`
- post-stall Cn surface: bias: 20 deg `0.0246621`, 45 deg `-0.0192779`, 70 deg `0.00980177`; beta: 20 deg `0.0107302`, 45 deg `-0.0462782`, 70 deg `0.07137`; p_hat: 20 deg `0.16538`, 45 deg `0.148062`, 70 deg `0.144007`; r_hat: 20 deg `0.487148`, 45 deg `0.00220622`, 70 deg `0.295961`
- post-stall Cmq residual: `0`
- selected post-stall surface replay scale: `0.250`
- transition blender status: `ok`
- transition blender fit group: `transition_before_post_stall`
- transition blender start alpha: `10.000` deg
- transition blender full alpha: `17.000` deg

## Replay Validation

- baseline train pitch MAE: `16.470` deg
- candidate train pitch MAE: `11.858` deg
- baseline held-out pitch MAE: `18.886` deg
- candidate held-out pitch MAE: `21.086` deg
- baseline held-out altitude-loss MAE: `0.3265` m
- candidate held-out altitude-loss MAE: `0.2650` m
- baseline held-out dx MAE: `0.7509` m
- candidate held-out dx MAE: `0.7613` m

## Stage Replay Errors

These are sample-aligned replay residuals grouped by the measured Vicon alpha regime. Use them to see whether the candidate fixes attached, transition, or post-stall behaviour; keep the full-throw held-out replay gate as the acceptance criterion.

- train/attached: samples `116`, throws `14`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.097` m, `0.009` m, `0.011` m, `0.297` m/s, `0.66` deg, `2.27` deg, `0.20` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.097` m, `0.009` m, `0.011` m, `0.295` m/s, `0.66` deg, `2.27` deg, `0.20` deg
- train/transition: samples `532`, throws `20`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.085` m, `0.100` m, `0.078` m, `0.388` m/s, `3.51` deg, `4.80` deg, `0.94` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.083` m, `0.080` m, `0.066` m, `0.345` m/s, `3.59` deg, `4.83` deg, `1.46` deg
- train/post_stall: samples `2851`, throws `20`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.350` m, `0.561` m, `0.273` m, `0.438` m/s, `23.30` deg, `10.20` deg, `9.47` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.334` m, `0.349` m, `0.212` m, `0.345` m/s, `26.73` deg, `11.44` deg, `17.05` deg
- heldout/attached: samples `65`, throws `8`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.019` m, `0.011` m, `0.010` m, `0.247` m/s, `0.54` deg, `1.63` deg, `0.31` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.019` m, `0.011` m, `0.010` m, `0.246` m/s, `0.54` deg, `1.63` deg, `0.31` deg
- heldout/transition: samples `265`, throws `10`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.209` m, `0.241` m, `0.184` m, `0.479` m/s, `7.09` deg, `9.90` deg, `3.75` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.205` m, `0.223` m, `0.174` m, `0.446` m/s, `7.11` deg, `10.02` deg, `3.03` deg
- heldout/post_stall: samples `1420`, throws `10`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.423` m, `0.766` m, `0.277` m, `0.412` m/s, `25.75` deg, `12.03` deg, `9.05` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.408` m, `0.540` m, `0.208` m, `0.320` m/s, `28.30` deg, `13.55` deg, `15.02` deg

## Candidate Parameters

- legacy scalar post-stall CL residual: baseline `0`, candidate `0`
- legacy scalar post-stall CD residual: baseline `0`, candidate `0`
- legacy scalar post-stall Cm residual: baseline `0`, candidate `0`
- post-stall CL surface: 20 deg baseline `0` -> candidate `0.0906696`, 45 deg baseline `0` -> candidate `-0.0848573`, 70 deg baseline `0` -> candidate `-0.0349095`
- post-stall CD surface: 20 deg baseline `0` -> candidate `0.0286968`, 45 deg baseline `0` -> candidate `6.27593e-05`, 70 deg baseline `0` -> candidate `-0.202031`
- post-stall Cm surface: 20 deg baseline `0` -> candidate `-0.00761777`, 45 deg baseline `0` -> candidate `-0.0196075`, 70 deg baseline `0` -> candidate `-0.00804303`
- post-stall Cmq surface: 20 deg baseline `0` -> candidate `1`, 45 deg baseline `0` -> candidate `-0.182242`, 70 deg baseline `0` -> candidate `1`
- post-stall CY surface: bias: 20 deg baseline `0` -> candidate `-0.0968507`, 45 deg baseline `0` -> candidate `0.104222`, 70 deg baseline `0` -> candidate `0.0705769`; beta: 20 deg baseline `0` -> candidate `-0.183259`, 45 deg baseline `0` -> candidate `-0.125313`, 70 deg baseline `0` -> candidate `-0.333186`; p_hat: 20 deg baseline `0` -> candidate `-0.527965`, 45 deg baseline `0` -> candidate `-0.170913`, 70 deg baseline `0` -> candidate `-0.1533`; r_hat: 20 deg baseline `0` -> candidate `-1`, 45 deg baseline `0` -> candidate `-0.287261`, 70 deg baseline `0` -> candidate `-1`
- post-stall Cl surface: bias: 20 deg baseline `0` -> candidate `-0.00216076`, 45 deg baseline `0` -> candidate `0.0011799`, 70 deg baseline `0` -> candidate `0.00874366`; beta: 20 deg baseline `0` -> candidate `0.00209127`, 45 deg baseline `0` -> candidate `-0.0171868`, 70 deg baseline `0` -> candidate `0.0209346`; p_hat: 20 deg baseline `0` -> candidate `0.0595225`, 45 deg baseline `0` -> candidate `0.100901`, 70 deg baseline `0` -> candidate `0.0758225`; r_hat: 20 deg baseline `0` -> candidate `-0.115367`, 45 deg baseline `0` -> candidate `0.0182426`, 70 deg baseline `0` -> candidate `-0.196003`
- post-stall Cn surface: bias: 20 deg baseline `0` -> candidate `0.00616551`, 45 deg baseline `0` -> candidate `-0.00481947`, 70 deg baseline `0` -> candidate `0.00245044`; beta: 20 deg baseline `0` -> candidate `0.00268256`, 45 deg baseline `0` -> candidate `-0.0115696`, 70 deg baseline `0` -> candidate `0.0178425`; p_hat: 20 deg baseline `0` -> candidate `0.0413449`, 45 deg baseline `0` -> candidate `0.0370155`, 70 deg baseline `0` -> candidate `0.0360018`; r_hat: 20 deg baseline `0` -> candidate `0.121787`, 45 deg baseline `0` -> candidate `0.000551556`, 70 deg baseline `0` -> candidate `0.0739901`
- baseline post-stall Cmq: `0`
- candidate post-stall Cmq: `0`
- baseline residual blend start alpha: `12.000` deg
- candidate residual blend start alpha: `10.000` deg
- baseline residual blend full alpha: `20.000` deg
- candidate residual blend full alpha: `17.000` deg

## Regime Summary

- train/attached: count `130`, Cm mean `0.02802`, Cm MAE `0.02929`, CY mean `-0.30905`, Cl mean `0.01304`, Cn mean `-0.00789`
- train/transition: count `538`, Cm mean `0.05084`, Cm MAE `0.06494`, CY mean `-0.26106`, Cl mean `-0.01008`, Cn mean `0.00038`
- train/post_stall: count `2851`, Cm mean `-0.15531`, Cm MAE `0.16575`, CY mean `0.54285`, Cl mean `-0.00275`, Cn mean `-0.00113`
- heldout/attached: count `73`, Cm mean `0.03199`, Cm MAE `0.03342`, CY mean `-0.34245`, Cl mean `0.01208`, Cn mean `-0.00877`
- heldout/transition: count `267`, Cm mean `0.00843`, Cm MAE `0.09653`, CY mean `-0.16047`, Cl mean `-0.02396`, Cn mean `0.00058`
- heldout/post_stall: count `1420`, Cm mean `-0.14198`, Cm MAE `0.15595`, CY mean `0.52116`, Cl mean `-0.00122`, Cn mean `0.00666`

## Independent Stage Fit Summary

Transition is split into samples before and after the first post-stall exposure in each throw. This separates real transition-model evidence from transition samples that may already be contaminated by earlier post-stall divergence.

- train/attached: count `130`, Cm mean `0.02802`, Cm MAE `0.02929`, Cm fit residual MAE `0.01845`, CY mean `-0.30905`, Cl mean `0.01304`, Cn mean `-0.00789`
- train/transition_before_post_stall: count `510`, Cm mean `0.05939`, Cm MAE `0.06275`, Cm fit residual MAE `0.03057`, CY mean `-0.32395`, Cl mean `-0.00930`, Cn mean `0.00113`
- train/transition_after_post_stall: count `28`, Cm mean `-0.10479`, Cm MAE `0.10479`, Cm fit residual MAE `0.09339`, CY mean `0.88438`, Cl mean `-0.02418`, Cn mean `-0.01323`
- train/post_stall: count `2851`, Cm mean `-0.15531`, Cm MAE `0.16575`, Cm fit residual MAE `0.06979`, CY mean `0.54285`, Cl mean `-0.00275`, Cn mean `-0.00113`
- heldout/attached: count `73`, Cm mean `0.03199`, Cm MAE `0.03342`, Cm fit residual MAE `0.02582`, CY mean `-0.34245`, Cl mean `0.01208`, Cn mean `-0.00877`
- heldout/transition_before_post_stall: count `210`, Cm mean `0.06454`, Cm MAE `0.06891`, Cm fit residual MAE `0.03301`, CY mean `-0.42131`, Cl mean `-0.01443`, Cn mean `0.00578`
- heldout/transition_after_post_stall: count `57`, Cm mean `-0.19830`, Cm MAE `0.19830`, Cm fit residual MAE `0.05693`, CY mean `0.80052`, Cl mean `-0.05906`, Cn mean `-0.01862`
- heldout/post_stall: count `1420`, Cm mean `-0.14198`, Cm MAE `0.15595`, Cm fit residual MAE `0.06659`, CY mean `0.52116`, Cl mean `-0.00122`, Cn mean `0.00666`

## Interpretation

Accept the candidate only if held-out replay improves pitch/q and lateral trajectory behaviour without damaging x, altitude loss, or sink. Attached and transition Cm residuals are diagnostic-only by default; accepted high-AoA changes should enter through the compact alpha-dependent post-stall static, pitch-damping, and lateral-directional residual surfaces.
