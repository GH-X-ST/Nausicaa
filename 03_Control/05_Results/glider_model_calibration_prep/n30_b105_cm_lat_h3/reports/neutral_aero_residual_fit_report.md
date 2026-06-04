# Neutral Aero Residual Regime Fit

This run uses only neutral open-loop real throws. It estimates 6-DoF force/moment residuals from Vicon state trajectories, keeps attached and transition Cm offsets report-only, fits compact post-stall alpha-RBF residual surfaces for CL/CD/Cm/Cmq plus CY/Cl/Cn lateral-directional coupling, then validates the candidate by held-out dry-air replay.

## Rerun Recipe

- source session root: `04_Flight_Test\05_Results\cal\n30\20260604_185934`
- alignment window: `0.100` s
- derivative window: `0.040` s
- replay dt: `0.0050` s
- ridge lambda: `0.001`
- min speed: `1.50` m/s
- workers: `8`
- apply attached Cm bias: `True`
- fit post-stall damping: `True`
- fit lateral surfaces: `True`

```powershell
python 03_Control/02_Inner_Loop/run_fit_neutral_aero_residual_calibration.py --session-root 04_Flight_Test\05_Results\cal\n30\20260604_185934 --run-label n30_b105_cm_lat_h3 --heldout-count 3 --heldout-seed 606 --alignment-window-s 0.1 --derivative-window-s 0.04 --replay-dt-s 0.005 --ridge-lambda 0.001 --min-speed-m-s 1.5 --workers 8 --apply-attached-cm-bias --fit-post-stall-damping --fit-lateral-surfaces
```

## Coefficient Fit

- fit status: `ok`
- sample count: `1010`
- used sample count: `1010`
- post-stall used sample count: `112`
- fit MAE in Cm: `0.04084`
- attached Cm residual: `0.068432`
- transition Cm residual before post-stall: `0.030888`
- transition Cm residual after post-stall: `0.0354749`
- post-stall surface centres: `20, 45, 70` deg
- post-stall surface width: `15` deg
- post-stall CL surface: 20 deg `0.644636`, 45 deg `-1.5`, 70 deg `0.589508`
- post-stall CD surface: 20 deg `0.235761`, 45 deg `-0.570223`, 70 deg `0.153788`
- post-stall Cm surface: 20 deg `0.0503704`, 45 deg `0.0410575`, 70 deg `0.174916`
- post-stall Cmq surface: 20 deg `2.82851`, 45 deg `0.5931`, 70 deg `0.00184462`
- post-stall CY surface: bias: 20 deg `-0.340975`, 45 deg `-0.964214`, 70 deg `-0.0394614`; beta: 20 deg `-1.50502`, 45 deg `-1.00113`, 70 deg `-0.0329385`; p_hat: 20 deg `1.28967`, 45 deg `-0.135341`, 70 deg `-0.0155798`; r_hat: 20 deg `2.41783`, 45 deg `0.45655`, 70 deg `0.00272357`
- post-stall Cl surface: bias: 20 deg `0.00766056`, 45 deg `-0.123541`, 70 deg `-0.00833297`; beta: 20 deg `0.00996757`, 45 deg `-0.0644117`, 70 deg `-0.00476405`; p_hat: 20 deg `0.34205`, 45 deg `0.043942`, 70 deg `-0.000619011`; r_hat: 20 deg `0.125899`, 45 deg `-0.01469`, 70 deg `-0.0019861`
- post-stall Cn surface: bias: 20 deg `0.0273526`, 45 deg `-0.0565662`, 70 deg `0.00677673`; beta: 20 deg `0.0469864`, 45 deg `0.092745`, 70 deg `-0.0015491`; p_hat: 20 deg `0.00931816`, 45 deg `0.00410816`, 70 deg `-0.00142871`; r_hat: 20 deg `0.290257`, 45 deg `0.0445929`, 70 deg `8.71657e-05`
- post-stall Cmq residual: `0`
- selected post-stall surface replay scale: `0.250`
- transition blender status: `ok`
- transition blender fit group: `transition_before_post_stall`
- transition blender start alpha: `13.000` deg
- transition blender full alpha: `20.000` deg

## Replay Validation

- baseline train pitch MAE: `19.411` deg
- candidate train pitch MAE: `10.625` deg
- baseline held-out pitch MAE: `14.376` deg
- candidate held-out pitch MAE: `9.695` deg
- baseline held-out altitude-loss MAE: `0.8964` m
- candidate held-out altitude-loss MAE: `0.3703` m
- baseline held-out dx MAE: `1.3825` m
- candidate held-out dx MAE: `1.1433` m

## Stage Replay Errors

These are sample-aligned replay residuals grouped by the measured Vicon alpha regime. Use them to see whether the candidate fixes attached, transition, or post-stall behaviour; keep the full-throw held-out replay gate as the acceptance criterion.

- train/attached: samples `201`, throws `7`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.242` m, `0.596` m, `0.292` m, `0.416` m/s, `8.55` deg, `9.19` deg, `6.62` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.160` m, `0.554` m, `0.100` m, `0.216` m/s, `12.14` deg, `6.11` deg, `10.82` deg
- train/transition: samples `690`, throws `7`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.272` m, `0.605` m, `0.457` m, `0.653` m/s, `7.53` deg, `18.58` deg, `7.51` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.195` m, `0.578` m, `0.237` m, `0.370` m/s, `9.74` deg, `9.66` deg, `11.20` deg
- train/post_stall: samples `112`, throws `2`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.178` m, `0.593` m, `0.168` m, `0.311` m/s, `6.30` deg, `9.92` deg, `5.11` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.076` m, `0.542` m, `0.084` m, `0.150` m/s, `8.91` deg, `12.75` deg, `12.40` deg
- heldout/attached: samples `29`, throws `3`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.121` m, `0.232` m, `0.141` m, `0.285` m/s, `3.35` deg, `4.79` deg, `5.30` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.079` m, `0.220` m, `0.022` m, `0.166` m/s, `10.83` deg, `1.16` deg, `10.17` deg
- heldout/transition: samples `344`, throws `3`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.796` m, `0.417` m, `0.428` m, `0.715` m/s, `10.35` deg, `11.88` deg, `7.99` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.679` m, `0.359` m, `0.267` m, `0.521` m/s, `16.75` deg, `4.50` deg, `13.68` deg
- heldout/post_stall: no replay samples

## Candidate Parameters

- legacy scalar post-stall CL residual: baseline `0`, candidate `0`
- legacy scalar post-stall CD residual: baseline `0`, candidate `0`
- legacy scalar post-stall Cm residual: baseline `0`, candidate `0`
- post-stall CL surface: 20 deg baseline `0` -> candidate `0.161159`, 45 deg baseline `0` -> candidate `-0.375`, 70 deg baseline `0` -> candidate `0.147377`
- post-stall CD surface: 20 deg baseline `0` -> candidate `0.0589402`, 45 deg baseline `0` -> candidate `-0.142556`, 70 deg baseline `0` -> candidate `0.0384469`
- post-stall Cm surface: 20 deg baseline `0` -> candidate `0.0125926`, 45 deg baseline `0` -> candidate `0.0102644`, 70 deg baseline `0` -> candidate `0.0437291`
- post-stall Cmq surface: 20 deg baseline `0` -> candidate `0.707127`, 45 deg baseline `0` -> candidate `0.148275`, 70 deg baseline `0` -> candidate `0.000461155`
- post-stall CY surface: bias: 20 deg baseline `0` -> candidate `-0.0852437`, 45 deg baseline `0` -> candidate `-0.241053`, 70 deg baseline `0` -> candidate `-0.00986534`; beta: 20 deg baseline `0` -> candidate `-0.376254`, 45 deg baseline `0` -> candidate `-0.250283`, 70 deg baseline `0` -> candidate `-0.00823461`; p_hat: 20 deg baseline `0` -> candidate `0.322418`, 45 deg baseline `0` -> candidate `-0.0338352`, 70 deg baseline `0` -> candidate `-0.00389495`; r_hat: 20 deg baseline `0` -> candidate `0.604458`, 45 deg baseline `0` -> candidate `0.114137`, 70 deg baseline `0` -> candidate `0.000680893`
- post-stall Cl surface: bias: 20 deg baseline `0` -> candidate `0.00191514`, 45 deg baseline `0` -> candidate `-0.0308851`, 70 deg baseline `0` -> candidate `-0.00208324`; beta: 20 deg baseline `0` -> candidate `0.00249189`, 45 deg baseline `0` -> candidate `-0.0161029`, 70 deg baseline `0` -> candidate `-0.00119101`; p_hat: 20 deg baseline `0` -> candidate `0.0855124`, 45 deg baseline `0` -> candidate `0.0109855`, 70 deg baseline `0` -> candidate `-0.000154753`; r_hat: 20 deg baseline `0` -> candidate `0.0314747`, 45 deg baseline `0` -> candidate `-0.00367249`, 70 deg baseline `0` -> candidate `-0.000496524`
- post-stall Cn surface: bias: 20 deg baseline `0` -> candidate `0.00683814`, 45 deg baseline `0` -> candidate `-0.0141416`, 70 deg baseline `0` -> candidate `0.00169418`; beta: 20 deg baseline `0` -> candidate `0.0117466`, 45 deg baseline `0` -> candidate `0.0231862`, 70 deg baseline `0` -> candidate `-0.000387275`; p_hat: 20 deg baseline `0` -> candidate `0.00232954`, 45 deg baseline `0` -> candidate `0.00102704`, 70 deg baseline `0` -> candidate `-0.000357178`; r_hat: 20 deg baseline `0` -> candidate `0.0725642`, 45 deg baseline `0` -> candidate `0.0111482`, 70 deg baseline `0` -> candidate `2.17914e-05`
- baseline post-stall Cmq: `0`
- candidate post-stall Cmq: `0`
- baseline residual blend start alpha: `12.000` deg
- candidate residual blend start alpha: `13.000` deg
- baseline residual blend full alpha: `20.000` deg
- candidate residual blend full alpha: `20.000` deg

## Regime Summary

- train/attached: count `207`, Cm mean `0.05525`, Cm MAE `0.07044`, CY mean `-0.18829`, Cl mean `0.00973`, Cn mean `-0.00134`
- train/transition: count `691`, Cm mean `0.03287`, Cm MAE `0.04434`, CY mean `-0.12766`, Cl mean `-0.01114`, Cn mean `0.00738`
- train/post_stall: count `112`, Cm mean `-0.00342`, Cm MAE `0.03669`, CY mean `-0.30715`, Cl mean `-0.05272`, Cn mean `0.01286`
- heldout/attached: count `32`, Cm mean `0.06851`, Cm MAE `0.06990`, CY mean `-0.27676`, Cl mean `0.00697`, Cn mean `-0.00519`
- heldout/transition: count `344`, Cm mean `0.03256`, Cm MAE `0.04260`, CY mean `-0.01824`, Cl mean `-0.00785`, Cn mean `-0.00034`
- heldout/post_stall: count `0`, Cm mean `nan`, Cm MAE `nan`, CY mean `nan`, Cl mean `nan`, Cn mean `nan`

## Independent Stage Fit Summary

Transition is split into samples before and after the first post-stall exposure in each throw. This separates real transition-model evidence from transition samples that may already be contaminated by earlier post-stall divergence.

- train/attached: count `207`, Cm mean `0.05525`, Cm MAE `0.07044`, Cm fit residual MAE `0.04732`, CY mean `-0.18829`, Cl mean `0.00973`, Cn mean `-0.00134`
- train/transition_before_post_stall: count `604`, Cm mean `0.03431`, Cm MAE `0.04488`, Cm fit residual MAE `0.04024`, CY mean `-0.16614`, Cl mean `-0.01058`, Cn mean `0.00831`
- train/transition_after_post_stall: count `87`, Cm mean `0.02287`, Cm MAE `0.04059`, Cm fit residual MAE `0.04154`, CY mean `0.13947`, Cl mean `-0.01506`, Cn mean `0.00090`
- train/post_stall: count `112`, Cm mean `-0.00342`, Cm MAE `0.03669`, Cm fit residual MAE `0.03150`, CY mean `-0.30715`, Cl mean `-0.05272`, Cn mean `0.01286`
- heldout/attached: count `32`, Cm mean `0.06851`, Cm MAE `0.06990`, Cm fit residual MAE `0.02571`, CY mean `-0.27676`, Cl mean `0.00697`, Cn mean `-0.00519`
- heldout/transition_before_post_stall: count `344`, Cm mean `0.03256`, Cm MAE `0.04260`, Cm fit residual MAE `0.04293`, CY mean `-0.01824`, Cl mean `-0.00785`, Cn mean `-0.00034`
- heldout/transition_after_post_stall: count `0`, Cm mean `nan`, Cm MAE `nan`, Cm fit residual MAE `nan`, CY mean `nan`, Cl mean `nan`, Cn mean `nan`
- heldout/post_stall: count `0`, Cm mean `nan`, Cm MAE `nan`, Cm fit residual MAE `nan`, CY mean `nan`, Cl mean `nan`, Cn mean `nan`

## Interpretation

Accept the candidate only if held-out replay improves pitch/q and lateral trajectory behaviour without damaging x, altitude loss, or sink. Attached and transition Cm residuals are diagnostic-only by default; accepted high-AoA changes should enter through the compact alpha-dependent post-stall static, pitch-damping, and lateral-directional residual surfaces.
