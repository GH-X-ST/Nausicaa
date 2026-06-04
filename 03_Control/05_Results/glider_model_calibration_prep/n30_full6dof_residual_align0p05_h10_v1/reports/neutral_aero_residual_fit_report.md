# Neutral Aero Residual Regime Fit

This run uses only neutral open-loop real throws. It estimates 6-DoF force/moment residuals from Vicon state trajectories, keeps attached and transition Cm offsets report-only, fits compact post-stall alpha-RBF residual surfaces for CL/CD/Cm/Cmq plus CY/Cl/Cn lateral-directional coupling, then validates the candidate by held-out dry-air replay.

## Rerun Recipe

- source session root: `04_Flight_Test\05_Results\cal\n30`
- alignment window: `0.050` s
- derivative window: `0.040` s
- replay dt: `0.0050` s
- ridge lambda: `0.001`
- min speed: `1.50` m/s
- workers: `8`
- apply attached Cm bias: `False`
- fit post-stall damping: `True`

```powershell
python 03_Control/02_Inner_Loop/run_fit_neutral_aero_residual_calibration.py --session-root 04_Flight_Test\05_Results\cal\n30 --run-label n30_full6dof_residual_align0p05_h10_v1 --heldout-count 10 --heldout-seed 606 --alignment-window-s 0.05 --derivative-window-s 0.04 --replay-dt-s 0.005 --ridge-lambda 0.001 --min-speed-m-s 1.5 --workers 8 --no-apply-attached-cm-bias --fit-post-stall-damping
```

## Coefficient Fit

- fit status: `ok`
- sample count: `3670`
- used sample count: `3670`
- post-stall used sample count: `2805`
- fit MAE in Cm: `0.07286`
- attached Cm residual: `0.028311`
- transition Cm residual before post-stall: `0.0573058`
- transition Cm residual after post-stall: `-0.105133`
- post-stall surface centres: `20, 45, 70` deg
- post-stall surface width: `15` deg
- post-stall CL surface: 20 deg `0.362687`, 45 deg `-0.33944`, 70 deg `-0.139633`
- post-stall CD surface: 20 deg `0.11478`, 45 deg `0.000265752`, 70 deg `-0.808156`
- post-stall Cm surface: 20 deg `-0.0215015`, 45 deg `-0.103347`, 70 deg `0.0311844`
- post-stall Cmq surface: 20 deg `4`, 45 deg `-1.7486`, 70 deg `4`
- post-stall CY surface: bias: 20 deg `-0.388327`, 45 deg `0.418642`, 70 deg `0.276371`; beta: 20 deg `-0.724394`, 45 deg `-0.51099`, 70 deg `-1.31627`; p_hat: 20 deg `-2.18538`, 45 deg `-0.625286`, 70 deg `-0.692649`; r_hat: 20 deg `-4`, 45 deg `-1.01807`, 70 deg `-4`
- post-stall Cl surface: bias: 20 deg `-0.00874072`, 45 deg `0.00496249`, 70 deg `0.0341465`; beta: 20 deg `0.00933693`, 45 deg `-0.0699812`, 70 deg `0.0856446`; p_hat: 20 deg `0.23092`, 45 deg `0.410389`, 70 deg `0.294307`; r_hat: 20 deg `-0.478525`, 45 deg `0.0894569`, 70 deg `-0.812151`
- post-stall Cn surface: bias: 20 deg `0.0247355`, 45 deg `-0.0194151`, 70 deg `0.010143`; beta: 20 deg `0.0099628`, 45 deg `-0.0451861`, 70 deg `0.0698736`; p_hat: 20 deg `0.171246`, 45 deg `0.143126`, 70 deg `0.150227`; r_hat: 20 deg `0.501533`, 45 deg `-0.0113876`, 70 deg `0.313454`
- post-stall Cmq residual: `0`
- selected post-stall surface replay scale: `0.250`
- transition blender status: `ok`
- transition blender fit group: `transition_before_post_stall`
- transition blender start alpha: `10.000` deg
- transition blender full alpha: `17.000` deg

## Replay Validation

- baseline train pitch MAE: `18.193` deg
- candidate train pitch MAE: `13.355` deg
- baseline held-out pitch MAE: `22.929` deg
- candidate held-out pitch MAE: `23.824` deg
- baseline held-out altitude-loss MAE: `0.8389` m
- candidate held-out altitude-loss MAE: `0.8187` m
- baseline held-out dx MAE: `1.0189` m
- candidate held-out dx MAE: `1.0112` m

## Stage Replay Errors

These are sample-aligned replay residuals grouped by the measured Vicon alpha regime. Use them to see whether the candidate fixes attached, transition, or post-stall behaviour; keep the full-throw held-out replay gate as the acceptance criterion.

- train/attached: samples `247`, throws `20`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.023` m, `0.010` m, `0.012` m, `0.160` m/s, `1.05` deg, `2.42` deg, `0.24` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.023` m, `0.010` m, `0.012` m, `0.160` m/s, `1.05` deg, `2.42` deg, `0.24` deg
- train/transition: samples `552`, throws `20`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.059` m, `0.119` m, `0.098` m, `0.378` m/s, `4.40` deg, `7.87` deg, `1.83` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.059` m, `0.101` m, `0.088` m, `0.354` m/s, `4.47` deg, `8.01` deg, `2.53` deg
- train/post_stall: samples `2851`, throws `20`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.403` m, `0.599` m, `0.391` m, `0.544` m/s, `23.46` deg, `13.33` deg, `10.93` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.397` m, `0.415` m, `0.344` m, `0.480` m/s, `26.75` deg, `13.97` deg, `18.67` deg
- heldout/attached: samples `127`, throws `9`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.019` m, `0.012` m, `0.012` m, `0.158` m/s, `1.10` deg, `2.96` deg, `0.33` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.019` m, `0.012` m, `0.012` m, `0.157` m/s, `1.10` deg, `2.96` deg, `0.33` deg
- heldout/transition: samples `276`, throws `10`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.262` m, `0.232` m, `0.329` m, `0.677` m/s, `7.01` deg, `15.80` deg, `4.12` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.259` m, `0.214` m, `0.321` m, `0.655` m/s, `6.88` deg, `15.86` deg, `4.04` deg
- heldout/post_stall: samples `1420`, throws `10`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.549` m, `0.712` m, `0.550` m, `0.755` m/s, `22.13` deg, `17.47` deg, `7.87` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.543` m, `0.536` m, `0.510` m, `0.704` m/s, `25.08` deg, `18.59` deg, `12.60` deg

## Candidate Parameters

- legacy scalar post-stall CL residual: baseline `0`, candidate `0`
- legacy scalar post-stall CD residual: baseline `0`, candidate `0`
- legacy scalar post-stall Cm residual: baseline `0`, candidate `0`
- post-stall CL surface: 20 deg baseline `0` -> candidate `0.0906717`, 45 deg baseline `0` -> candidate `-0.08486`, 70 deg baseline `0` -> candidate `-0.0349082`
- post-stall CD surface: 20 deg baseline `0` -> candidate `0.0286951`, 45 deg baseline `0` -> candidate `6.6438e-05`, 70 deg baseline `0` -> candidate `-0.202039`
- post-stall Cm surface: 20 deg baseline `0` -> candidate `-0.00537539`, 45 deg baseline `0` -> candidate `-0.0258367`, 70 deg baseline `0` -> candidate `0.00779611`
- post-stall Cmq surface: 20 deg baseline `0` -> candidate `1`, 45 deg baseline `0` -> candidate `-0.437151`, 70 deg baseline `0` -> candidate `1`
- post-stall CY surface: bias: 20 deg baseline `0` -> candidate `-0.0970818`, 45 deg baseline `0` -> candidate `0.10466`, 70 deg baseline `0` -> candidate `0.0690928`; beta: 20 deg baseline `0` -> candidate `-0.181098`, 45 deg baseline `0` -> candidate `-0.127747`, 70 deg baseline `0` -> candidate `-0.329067`; p_hat: 20 deg baseline `0` -> candidate `-0.546345`, 45 deg baseline `0` -> candidate `-0.156321`, 70 deg baseline `0` -> candidate `-0.173162`; r_hat: 20 deg baseline `0` -> candidate `-1`, 45 deg baseline `0` -> candidate `-0.254517`, 70 deg baseline `0` -> candidate `-1`
- post-stall Cl surface: bias: 20 deg baseline `0` -> candidate `-0.00218518`, 45 deg baseline `0` -> candidate `0.00124062`, 70 deg baseline `0` -> candidate `0.00853662`; beta: 20 deg baseline `0` -> candidate `0.00233423`, 45 deg baseline `0` -> candidate `-0.0174953`, 70 deg baseline `0` -> candidate `0.0214111`; p_hat: 20 deg baseline `0` -> candidate `0.05773`, 45 deg baseline `0` -> candidate `0.102597`, 70 deg baseline `0` -> candidate `0.0735768`; r_hat: 20 deg baseline `0` -> candidate `-0.119631`, 45 deg baseline `0` -> candidate `0.0223642`, 70 deg baseline `0` -> candidate `-0.203038`
- post-stall Cn surface: bias: 20 deg baseline `0` -> candidate `0.00618388`, 45 deg baseline `0` -> candidate `-0.00485377`, 70 deg baseline `0` -> candidate `0.00253576`; beta: 20 deg baseline `0` -> candidate `0.0024907`, 45 deg baseline `0` -> candidate `-0.0112965`, 70 deg baseline `0` -> candidate `0.0174684`; p_hat: 20 deg baseline `0` -> candidate `0.0428114`, 45 deg baseline `0` -> candidate `0.0357814`, 70 deg baseline `0` -> candidate `0.0375568`; r_hat: 20 deg baseline `0` -> candidate `0.125383`, 45 deg baseline `0` -> candidate `-0.0028469`, 70 deg baseline `0` -> candidate `0.0783634`
- baseline post-stall Cmq: `0`
- candidate post-stall Cmq: `0`
- baseline residual blend start alpha: `12.000` deg
- candidate residual blend start alpha: `10.000` deg
- baseline residual blend full alpha: `20.000` deg
- candidate residual blend full alpha: `17.000` deg

## Regime Summary

- train/attached: count `267`, Cm mean `0.04088`, Cm MAE `0.04162`, CY mean `-0.26195`, Cl mean `0.01362`, Cn mean `-0.00957`
- train/transition: count `552`, Cm mean `0.05033`, Cm MAE `0.06411`, CY mean `-0.26311`, Cl mean `-0.00973`, Cn mean `0.00015`
- train/post_stall: count `2851`, Cm mean `-0.15531`, Cm MAE `0.16575`, CY mean `0.54285`, Cl mean `-0.00275`, Cn mean `-0.00113`
- heldout/attached: count `136`, Cm mean `0.05732`, Cm MAE `0.05809`, CY mean `-0.29479`, Cl mean `0.01641`, Cn mean `-0.01016`
- heldout/transition: count `277`, Cm mean `0.01149`, Cm MAE `0.09641`, CY mean `-0.16923`, Cl mean `-0.02323`, Cn mean `0.00044`
- heldout/post_stall: count `1420`, Cm mean `-0.14198`, Cm MAE `0.15595`, CY mean `0.52116`, Cl mean `-0.00122`, Cn mean `0.00666`

## Independent Stage Fit Summary

Transition is split into samples before and after the first post-stall exposure in each throw. This separates real transition-model evidence from transition samples that may already be contaminated by earlier post-stall divergence.

- train/attached: count `267`, Cm mean `0.04088`, Cm MAE `0.04162`, Cm fit residual MAE `0.02925`, CY mean `-0.26195`, Cl mean `0.01362`, Cn mean `-0.00957`
- train/transition_before_post_stall: count `524`, Cm mean `0.05861`, Cm MAE `0.06194`, Cm fit residual MAE `0.03042`, CY mean `-0.32443`, Cl mean `-0.00896`, Cn mean `0.00087`
- train/transition_after_post_stall: count `28`, Cm mean `-0.10479`, Cm MAE `0.10479`, Cm fit residual MAE `0.08098`, CY mean `0.88438`, Cl mean `-0.02418`, Cn mean `-0.01323`
- train/post_stall: count `2851`, Cm mean `-0.15531`, Cm MAE `0.16575`, Cm fit residual MAE `0.08466`, CY mean `0.54285`, Cl mean `-0.00275`, Cn mean `-0.00113`
- heldout/attached: count `136`, Cm mean `0.05732`, Cm MAE `0.05809`, Cm fit residual MAE `0.04291`, CY mean `-0.29479`, Cl mean `0.01641`, Cn mean `-0.01016`
- heldout/transition_before_post_stall: count `220`, Cm mean `0.06584`, Cm MAE `0.07001`, Cm fit residual MAE `0.03439`, CY mean `-0.42048`, Cl mean `-0.01394`, Cn mean `0.00537`
- heldout/transition_after_post_stall: count `57`, Cm mean `-0.19830`, Cm MAE `0.19830`, Cm fit residual MAE `0.05947`, CY mean `0.80052`, Cl mean `-0.05906`, Cn mean `-0.01862`
- heldout/post_stall: count `1420`, Cm mean `-0.14198`, Cm MAE `0.15595`, Cm fit residual MAE `0.08189`, CY mean `0.52116`, Cl mean `-0.00122`, Cn mean `0.00666`

## Interpretation

Accept the candidate only if held-out replay improves pitch/q and lateral trajectory behaviour without damaging x, altitude loss, or sink. Attached and transition Cm residuals are diagnostic-only by default; accepted high-AoA changes should enter through the compact alpha-dependent post-stall static, pitch-damping, and lateral-directional residual surfaces.
