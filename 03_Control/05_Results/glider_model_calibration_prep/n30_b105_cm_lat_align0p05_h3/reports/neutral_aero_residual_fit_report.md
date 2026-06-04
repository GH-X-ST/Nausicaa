# Neutral Aero Residual Regime Fit

This run uses only neutral open-loop real throws. It estimates 6-DoF force/moment residuals from Vicon state trajectories, keeps attached and transition Cm offsets report-only, fits compact post-stall alpha-RBF residual surfaces for CL/CD/Cm/Cmq plus CY/Cl/Cn lateral-directional coupling, then validates the candidate by held-out dry-air replay.

## Rerun Recipe

- source session root: `04_Flight_Test\05_Results\cal\n30\20260604_185934`
- alignment window: `0.050` s
- derivative window: `0.040` s
- replay dt: `0.0050` s
- ridge lambda: `0.001`
- min speed: `1.50` m/s
- workers: `8`
- apply attached Cm bias: `True`
- fit post-stall damping: `True`
- fit lateral surfaces: `True`

```powershell
python 03_Control/02_Inner_Loop/run_fit_neutral_aero_residual_calibration.py --session-root 04_Flight_Test\05_Results\cal\n30\20260604_185934 --run-label n30_b105_cm_lat_align0p05_h3 --heldout-count 3 --heldout-seed 606 --alignment-window-s 0.05 --derivative-window-s 0.04 --replay-dt-s 0.005 --ridge-lambda 0.001 --min-speed-m-s 1.5 --workers 8 --apply-attached-cm-bias --fit-post-stall-damping --fit-lateral-surfaces
```

## Coefficient Fit

- fit status: `ok`
- sample count: `1056`
- used sample count: `1056`
- post-stall used sample count: `112`
- fit MAE in Cm: `0.04359`
- attached Cm residual: `0.0837469`
- transition Cm residual before post-stall: `0.0310405`
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
- selected post-stall surface replay scale: `0.500`
- transition blender status: `ok`
- transition blender fit group: `transition_before_post_stall`
- transition blender start alpha: `13.000` deg
- transition blender full alpha: `20.000` deg

## Replay Validation

- baseline train pitch MAE: `26.026` deg
- candidate train pitch MAE: `14.709` deg
- baseline held-out pitch MAE: `18.582` deg
- candidate held-out pitch MAE: `19.020` deg
- baseline held-out altitude-loss MAE: `1.3040` m
- candidate held-out altitude-loss MAE: `0.4717` m
- baseline held-out dx MAE: `1.6146` m
- candidate held-out dx MAE: `1.2913` m

## Stage Replay Errors

These are sample-aligned replay residuals grouped by the measured Vicon alpha regime. Use them to see whether the candidate fixes attached, transition, or post-stall behaviour; keep the full-throw held-out replay gate as the acceptance criterion.

- train/attached: samples `245`, throws `7`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.208` m, `0.509` m, `0.396` m, `0.510` m/s, `7.52` deg, `11.86` deg, `5.48` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.116` m, `0.460` m, `0.106` m, `0.209` m/s, `10.67` deg, `7.58` deg, `12.25` deg
- train/transition: samples `692`, throws `7`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.277` m, `0.629` m, `0.701` m, `0.933` m/s, `7.52` deg, `25.21` deg, `7.49` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.205` m, `0.562` m, `0.301` m, `0.437` m/s, `8.82` deg, `11.82` deg, `13.19` deg
- train/post_stall: samples `112`, throws `2`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.167` m, `0.676` m, `0.302` m, `0.505` m/s, `15.62` deg, `12.99` deg, `5.96` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.046` m, `0.558` m, `0.122` m, `0.205` m/s, `11.86` deg, `14.19` deg, `19.23` deg
- heldout/attached: samples `41`, throws `3`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.080` m, `0.162` m, `0.124` m, `0.233` m/s, `3.10` deg, `4.92` deg, `3.56` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.109` m, `0.063` m, `0.032` m, `0.138` m/s, `3.28` deg, `3.02` deg, `5.77` deg
- heldout/transition: samples `344`, throws `3`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.909` m, `0.402` m, `0.582` m, `0.885` m/s, `10.60` deg, `15.11` deg, `7.37` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.776` m, `0.342` m, `0.290` m, `0.526` m/s, `18.51` deg, `7.88` deg, `19.96` deg
- heldout/post_stall: no replay samples

## Candidate Parameters

- legacy scalar post-stall CL residual: baseline `0`, candidate `0`
- legacy scalar post-stall CD residual: baseline `0`, candidate `0`
- legacy scalar post-stall Cm residual: baseline `0`, candidate `0`
- post-stall CL surface: 20 deg baseline `0` -> candidate `0.322318`, 45 deg baseline `0` -> candidate `-0.75`, 70 deg baseline `0` -> candidate `0.294754`
- post-stall CD surface: 20 deg baseline `0` -> candidate `0.11788`, 45 deg baseline `0` -> candidate `-0.285111`, 70 deg baseline `0` -> candidate `0.0768938`
- post-stall Cm surface: 20 deg baseline `0` -> candidate `0.0251852`, 45 deg baseline `0` -> candidate `0.0205288`, 70 deg baseline `0` -> candidate `0.0874582`
- post-stall Cmq surface: 20 deg baseline `0` -> candidate `1.41425`, 45 deg baseline `0` -> candidate `0.29655`, 70 deg baseline `0` -> candidate `0.000922311`
- post-stall CY surface: bias: 20 deg baseline `0` -> candidate `-0.170487`, 45 deg baseline `0` -> candidate `-0.482107`, 70 deg baseline `0` -> candidate `-0.0197307`; beta: 20 deg baseline `0` -> candidate `-0.752508`, 45 deg baseline `0` -> candidate `-0.500565`, 70 deg baseline `0` -> candidate `-0.0164692`; p_hat: 20 deg baseline `0` -> candidate `0.644835`, 45 deg baseline `0` -> candidate `-0.0676705`, 70 deg baseline `0` -> candidate `-0.0077899`; r_hat: 20 deg baseline `0` -> candidate `1.20892`, 45 deg baseline `0` -> candidate `0.228275`, 70 deg baseline `0` -> candidate `0.00136179`
- post-stall Cl surface: bias: 20 deg baseline `0` -> candidate `0.00383028`, 45 deg baseline `0` -> candidate `-0.0617703`, 70 deg baseline `0` -> candidate `-0.00416648`; beta: 20 deg baseline `0` -> candidate `0.00498378`, 45 deg baseline `0` -> candidate `-0.0322059`, 70 deg baseline `0` -> candidate `-0.00238203`; p_hat: 20 deg baseline `0` -> candidate `0.171025`, 45 deg baseline `0` -> candidate `0.021971`, 70 deg baseline `0` -> candidate `-0.000309506`; r_hat: 20 deg baseline `0` -> candidate `0.0629495`, 45 deg baseline `0` -> candidate `-0.00734499`, 70 deg baseline `0` -> candidate `-0.000993049`
- post-stall Cn surface: bias: 20 deg baseline `0` -> candidate `0.0136763`, 45 deg baseline `0` -> candidate `-0.0282831`, 70 deg baseline `0` -> candidate `0.00338836`; beta: 20 deg baseline `0` -> candidate `0.0234932`, 45 deg baseline `0` -> candidate `0.0463725`, 70 deg baseline `0` -> candidate `-0.00077455`; p_hat: 20 deg baseline `0` -> candidate `0.00465908`, 45 deg baseline `0` -> candidate `0.00205408`, 70 deg baseline `0` -> candidate `-0.000714356`; r_hat: 20 deg baseline `0` -> candidate `0.145128`, 45 deg baseline `0` -> candidate `0.0222965`, 70 deg baseline `0` -> candidate `4.35828e-05`
- baseline post-stall Cmq: `0`
- candidate post-stall Cmq: `0`
- baseline residual blend start alpha: `12.000` deg
- candidate residual blend start alpha: `13.000` deg
- baseline residual blend full alpha: `20.000` deg
- candidate residual blend full alpha: `20.000` deg

## Regime Summary

- train/attached: count `252`, Cm mean `0.07196`, Cm MAE `0.08444`, CY mean `-0.19963`, Cl mean `0.01342`, Cn mean `-0.00444`
- train/transition: count `692`, Cm mean `0.03289`, Cm MAE `0.04434`, CY mean `-0.12796`, Cl mean `-0.01112`, Cn mean `0.00734`
- train/post_stall: count `112`, Cm mean `-0.00342`, Cm MAE `0.03669`, CY mean `-0.30715`, Cl mean `-0.05272`, Cn mean `0.01286`
- heldout/attached: count `44`, Cm mean `0.07370`, Cm MAE `0.07471`, CY mean `-0.26063`, Cl mean `0.01141`, Cn mean `-0.00693`
- heldout/transition: count `344`, Cm mean `0.03256`, Cm MAE `0.04260`, CY mean `-0.01824`, Cl mean `-0.00785`, Cn mean `-0.00034`
- heldout/post_stall: count `0`, Cm mean `nan`, Cm MAE `nan`, CY mean `nan`, Cl mean `nan`, Cn mean `nan`

## Independent Stage Fit Summary

Transition is split into samples before and after the first post-stall exposure in each throw. This separates real transition-model evidence from transition samples that may already be contaminated by earlier post-stall divergence.

- train/attached: count `252`, Cm mean `0.07196`, Cm MAE `0.08444`, Cm fit residual MAE `0.05767`, CY mean `-0.19963`, Cl mean `0.01342`, Cn mean `-0.00444`
- train/transition_before_post_stall: count `605`, Cm mean `0.03433`, Cm MAE `0.04488`, Cm fit residual MAE `0.04025`, CY mean `-0.16642`, Cl mean `-0.01055`, Cn mean `0.00827`
- train/transition_after_post_stall: count `87`, Cm mean `0.02287`, Cm MAE `0.04059`, Cm fit residual MAE `0.04154`, CY mean `0.13947`, Cl mean `-0.01506`, Cn mean `0.00090`
- train/post_stall: count `112`, Cm mean `-0.00342`, Cm MAE `0.03669`, Cm fit residual MAE `0.03150`, CY mean `-0.30715`, Cl mean `-0.05272`, Cn mean `0.01286`
- heldout/attached: count `44`, Cm mean `0.07370`, Cm MAE `0.07471`, Cm fit residual MAE `0.02994`, CY mean `-0.26063`, Cl mean `0.01141`, Cn mean `-0.00693`
- heldout/transition_before_post_stall: count `344`, Cm mean `0.03256`, Cm MAE `0.04260`, Cm fit residual MAE `0.04293`, CY mean `-0.01824`, Cl mean `-0.00785`, Cn mean `-0.00034`
- heldout/transition_after_post_stall: count `0`, Cm mean `nan`, Cm MAE `nan`, Cm fit residual MAE `nan`, CY mean `nan`, Cl mean `nan`, Cn mean `nan`
- heldout/post_stall: count `0`, Cm mean `nan`, Cm MAE `nan`, Cm fit residual MAE `nan`, CY mean `nan`, Cl mean `nan`, Cn mean `nan`

## Interpretation

Accept the candidate only if held-out replay improves pitch/q and lateral trajectory behaviour without damaging x, altitude loss, or sink. Attached and transition Cm residuals are diagnostic-only by default; accepted high-AoA changes should enter through the compact alpha-dependent post-stall static, pitch-damping, and lateral-directional residual surfaces.
