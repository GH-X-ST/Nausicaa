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
- apply attached Cm bias: `False`
- fit post-stall damping: `False`

```powershell
python 03_Control/02_Inner_Loop/run_fit_neutral_aero_residual_calibration.py --session-root 04_Flight_Test\05_Results\cal\n30 --run-label n30_full6dof_static_surface_h10_v1 --heldout-count 10 --heldout-seed 606 --alignment-window-s 0.1 --derivative-window-s 0.04 --replay-dt-s 0.005 --ridge-lambda 0.001 --min-speed-m-s 1.5 --workers 8 --no-apply-attached-cm-bias --no-fit-post-stall-damping
```

## Coefficient Fit

- fit status: `ok`
- sample count: `3519`
- used sample count: `3519`
- post-stall used sample count: `2805`
- fit MAE in Cm: `0.07799`
- attached Cm residual: `0.0191496`
- transition Cm residual before post-stall: `0.0588549`
- transition Cm residual after post-stall: `-0.105133`
- post-stall surface centres: `20, 45, 70` deg
- post-stall surface width: `15` deg
- post-stall CL surface: 20 deg `0.362687`, 45 deg `-0.33944`, 70 deg `-0.139633`
- post-stall CD surface: 20 deg `0.11478`, 45 deg `0.000265752`, 70 deg `-0.808156`
- post-stall Cm surface: 20 deg `-0.00410382`, 45 deg `-0.141443`, 70 deg `-0.218001`
- post-stall Cmq surface: 20 deg `0`, 45 deg `0`, 70 deg `0`
- post-stall CY surface: bias: 20 deg `-0.388327`, 45 deg `0.418642`, 70 deg `0.276371`; beta: 20 deg `-0.724394`, 45 deg `-0.51099`, 70 deg `-1.31627`; p_hat: 20 deg `-2.18538`, 45 deg `-0.625286`, 70 deg `-0.692649`; r_hat: 20 deg `-4`, 45 deg `-1.01807`, 70 deg `-4`
- post-stall Cl surface: bias: 20 deg `-0.00874072`, 45 deg `0.00496249`, 70 deg `0.0341465`; beta: 20 deg `0.00933693`, 45 deg `-0.0699812`, 70 deg `0.0856446`; p_hat: 20 deg `0.23092`, 45 deg `0.410389`, 70 deg `0.294307`; r_hat: 20 deg `-0.478525`, 45 deg `0.0894569`, 70 deg `-0.812151`
- post-stall Cn surface: bias: 20 deg `0.0247355`, 45 deg `-0.0194151`, 70 deg `0.010143`; beta: 20 deg `0.0099628`, 45 deg `-0.0451861`, 70 deg `0.0698736`; p_hat: 20 deg `0.171246`, 45 deg `0.143126`, 70 deg `0.150227`; r_hat: 20 deg `0.501533`, 45 deg `-0.0113876`, 70 deg `0.313454`
- post-stall Cmq residual: `0`
- selected post-stall surface replay scale: `0.250`
- transition blender status: `ok`
- transition blender fit group: `transition_before_post_stall`
- transition blender start alpha: `11.000` deg
- transition blender full alpha: `20.000` deg

## Replay Validation

- baseline train pitch MAE: `16.470` deg
- candidate train pitch MAE: `11.064` deg
- baseline held-out pitch MAE: `18.886` deg
- candidate held-out pitch MAE: `19.566` deg
- baseline held-out altitude-loss MAE: `0.3265` m
- candidate held-out altitude-loss MAE: `0.2666` m
- baseline held-out dx MAE: `0.7509` m
- candidate held-out dx MAE: `0.8281` m

## Stage Replay Errors

These are sample-aligned replay residuals grouped by the measured Vicon alpha regime. Use them to see whether the candidate fixes attached, transition, or post-stall behaviour; keep the full-throw held-out replay gate as the acceptance criterion.

- train/attached: samples `116`, throws `14`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.097` m, `0.009` m, `0.011` m, `0.297` m/s, `0.66` deg, `2.27` deg, `0.20` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.097` m, `0.009` m, `0.011` m, `0.296` m/s, `0.66` deg, `2.27` deg, `0.20` deg
- train/transition: samples `532`, throws `20`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.085` m, `0.100` m, `0.078` m, `0.388` m/s, `3.51` deg, `4.80` deg, `0.94` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.085` m, `0.084` m, `0.069` m, `0.362` m/s, `3.52` deg, `5.09` deg, `1.30` deg
- train/post_stall: samples `2851`, throws `20`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.350` m, `0.561` m, `0.273` m, `0.438` m/s, `23.30` deg, `10.20` deg, `9.47` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.361` m, `0.369` m, `0.222` m, `0.363` m/s, `26.16` deg, `11.66` deg, `17.21` deg
- heldout/attached: samples `65`, throws `8`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.019` m, `0.011` m, `0.010` m, `0.247` m/s, `0.54` deg, `1.63` deg, `0.31` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.019` m, `0.011` m, `0.010` m, `0.247` m/s, `0.54` deg, `1.63` deg, `0.31` deg
- heldout/transition: samples `265`, throws `10`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.209` m, `0.241` m, `0.184` m, `0.479` m/s, `7.09` deg, `9.90` deg, `3.75` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.207` m, `0.227` m, `0.177` m, `0.458` m/s, `7.06` deg, `10.20` deg, `3.05` deg
- heldout/post_stall: samples `1420`, throws `10`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.423` m, `0.766` m, `0.277` m, `0.412` m/s, `25.75` deg, `12.03` deg, `9.05` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.435` m, `0.567` m, `0.218` m, `0.336` m/s, `27.69` deg, `13.77` deg, `15.14` deg

## Candidate Parameters

- legacy scalar post-stall CL residual: baseline `0`, candidate `0`
- legacy scalar post-stall CD residual: baseline `0`, candidate `0`
- legacy scalar post-stall Cm residual: baseline `0`, candidate `0`
- post-stall CL surface: 20 deg baseline `0` -> candidate `0.0906717`, 45 deg baseline `0` -> candidate `-0.08486`, 70 deg baseline `0` -> candidate `-0.0349082`
- post-stall CD surface: 20 deg baseline `0` -> candidate `0.0286951`, 45 deg baseline `0` -> candidate `6.6438e-05`, 70 deg baseline `0` -> candidate `-0.202039`
- post-stall Cm surface: 20 deg baseline `0` -> candidate `-0.00102596`, 45 deg baseline `0` -> candidate `-0.0353607`, 70 deg baseline `0` -> candidate `-0.0545003`
- post-stall Cmq surface: 20 deg baseline `0` -> candidate `0`, 45 deg baseline `0` -> candidate `0`, 70 deg baseline `0` -> candidate `0`
- post-stall CY surface: bias: 20 deg baseline `0` -> candidate `-0.0970818`, 45 deg baseline `0` -> candidate `0.10466`, 70 deg baseline `0` -> candidate `0.0690928`; beta: 20 deg baseline `0` -> candidate `-0.181098`, 45 deg baseline `0` -> candidate `-0.127747`, 70 deg baseline `0` -> candidate `-0.329067`; p_hat: 20 deg baseline `0` -> candidate `-0.546345`, 45 deg baseline `0` -> candidate `-0.156321`, 70 deg baseline `0` -> candidate `-0.173162`; r_hat: 20 deg baseline `0` -> candidate `-1`, 45 deg baseline `0` -> candidate `-0.254517`, 70 deg baseline `0` -> candidate `-1`
- post-stall Cl surface: bias: 20 deg baseline `0` -> candidate `-0.00218518`, 45 deg baseline `0` -> candidate `0.00124062`, 70 deg baseline `0` -> candidate `0.00853662`; beta: 20 deg baseline `0` -> candidate `0.00233423`, 45 deg baseline `0` -> candidate `-0.0174953`, 70 deg baseline `0` -> candidate `0.0214111`; p_hat: 20 deg baseline `0` -> candidate `0.05773`, 45 deg baseline `0` -> candidate `0.102597`, 70 deg baseline `0` -> candidate `0.0735768`; r_hat: 20 deg baseline `0` -> candidate `-0.119631`, 45 deg baseline `0` -> candidate `0.0223642`, 70 deg baseline `0` -> candidate `-0.203038`
- post-stall Cn surface: bias: 20 deg baseline `0` -> candidate `0.00618388`, 45 deg baseline `0` -> candidate `-0.00485377`, 70 deg baseline `0` -> candidate `0.00253576`; beta: 20 deg baseline `0` -> candidate `0.0024907`, 45 deg baseline `0` -> candidate `-0.0112965`, 70 deg baseline `0` -> candidate `0.0174684`; p_hat: 20 deg baseline `0` -> candidate `0.0428114`, 45 deg baseline `0` -> candidate `0.0357814`, 70 deg baseline `0` -> candidate `0.0375568`; r_hat: 20 deg baseline `0` -> candidate `0.125383`, 45 deg baseline `0` -> candidate `-0.0028469`, 70 deg baseline `0` -> candidate `0.0783634`
- baseline post-stall Cmq: `0`
- candidate post-stall Cmq: `0`
- baseline residual blend start alpha: `12.000` deg
- candidate residual blend start alpha: `11.000` deg
- baseline residual blend full alpha: `20.000` deg
- candidate residual blend full alpha: `20.000` deg

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
- train/transition_before_post_stall: count `510`, Cm mean `0.05939`, Cm MAE `0.06275`, Cm fit residual MAE `0.07853`, CY mean `-0.32395`, Cl mean `-0.00930`, Cn mean `0.00113`
- train/transition_after_post_stall: count `28`, Cm mean `-0.10479`, Cm MAE `0.10479`, Cm fit residual MAE `0.07271`, CY mean `0.88438`, Cl mean `-0.02418`, Cn mean `-0.01323`
- train/post_stall: count `2851`, Cm mean `-0.15531`, Cm MAE `0.16575`, Cm fit residual MAE `0.08066`, CY mean `0.54285`, Cl mean `-0.00275`, Cn mean `-0.00113`
- heldout/attached: count `73`, Cm mean `0.03199`, Cm MAE `0.03342`, Cm fit residual MAE `0.02582`, CY mean `-0.34245`, Cl mean `0.01208`, Cn mean `-0.00877`
- heldout/transition_before_post_stall: count `210`, Cm mean `0.06454`, Cm MAE `0.06891`, Cm fit residual MAE `0.08513`, CY mean `-0.42131`, Cl mean `-0.01443`, Cn mean `0.00578`
- heldout/transition_after_post_stall: count `57`, Cm mean `-0.19830`, Cm MAE `0.19830`, Cm fit residual MAE `0.16712`, CY mean `0.80052`, Cl mean `-0.05906`, Cn mean `-0.01862`
- heldout/post_stall: count `1420`, Cm mean `-0.14198`, Cm MAE `0.15595`, Cm fit residual MAE `0.07037`, CY mean `0.52116`, Cl mean `-0.00122`, Cn mean `0.00666`

## Interpretation

Accept the candidate only if held-out replay improves pitch/q and lateral trajectory behaviour without damaging x, altitude loss, or sink. Attached and transition Cm residuals are diagnostic-only by default; accepted high-AoA changes should enter through the compact alpha-dependent post-stall static, pitch-damping, and lateral-directional residual surfaces.
