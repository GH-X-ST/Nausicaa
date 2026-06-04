# Neutral Aero Residual Regime Fit

This run uses only neutral open-loop real throws. It estimates force/moment residuals from Vicon state trajectories, keeps attached-flow corrections report-only, uses transition only as a smooth activation band, and validates post-stall residual candidates by held-out dry-air replay.

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
python 03_Control/02_Inner_Loop/run_fit_neutral_aero_residual_calibration.py --session-root 04_Flight_Test\05_Results\cal\n30 --run-label n30_transition_poststall_light_h10_stage_replay_v1 --heldout-count 10 --heldout-seed 606 --alignment-window-s 0.1 --derivative-window-s 0.04 --replay-dt-s 0.005 --ridge-lambda 0.001 --min-speed-m-s 1.5 --workers 8 --no-apply-attached-cm-bias --no-fit-post-stall-damping
```

## Coefficient Fit

- fit status: `ok`
- sample count: `3519`
- used sample count: `3519`
- fit MAE in Cm: `0.08903`
- attached Cm residual: `0.0305529`
- transition Cm residual: `0.112407`
- post-stall CL residual: `0.0540889`
- post-stall CD residual: `-0.0764659`
- post-stall Cm residual: `-0.124592`
- post-stall Cmq residual: `0`

## Replay Validation

- baseline train pitch MAE: `16.470` deg
- candidate train pitch MAE: `14.702` deg
- baseline held-out pitch MAE: `18.886` deg
- candidate held-out pitch MAE: `17.728` deg
- baseline held-out altitude-loss MAE: `0.3265` m
- candidate held-out altitude-loss MAE: `0.5640` m
- baseline held-out dx MAE: `0.7509` m
- candidate held-out dx MAE: `1.2625` m

## Stage Replay Errors

These are sample-aligned replay residuals grouped by the measured Vicon alpha regime. Use them to see whether the candidate fixes attached, transition, or post-stall behaviour; keep the full-throw held-out replay gate as the acceptance criterion.

- train/attached: samples `116`, throws `14`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.097` m, `0.009` m, `0.011` m, `0.297` m/s, `0.66` deg, `2.27` deg, `0.20` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.097` m, `0.009` m, `0.011` m, `0.297` m/s, `0.66` deg, `2.27` deg, `0.20` deg
- train/transition: samples `532`, throws `20`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.085` m, `0.100` m, `0.078` m, `0.388` m/s, `3.51` deg, `4.80` deg, `0.94` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.094` m, `0.103` m, `0.079` m, `0.384` m/s, `4.53` deg, `6.31` deg, `0.91` deg
- train/post_stall: samples `2851`, throws `20`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.350` m, `0.561` m, `0.273` m, `0.438` m/s, `23.30` deg, `10.20` deg, `9.47` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.570` m, `0.647` m, `0.335` m, `0.497` m/s, `34.12` deg, `19.09` deg, `10.13` deg
- heldout/attached: samples `65`, throws `8`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.019` m, `0.011` m, `0.010` m, `0.247` m/s, `0.54` deg, `1.63` deg, `0.31` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.019` m, `0.011` m, `0.010` m, `0.247` m/s, `0.54` deg, `1.63` deg, `0.31` deg
- heldout/transition: samples `265`, throws `10`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.209` m, `0.241` m, `0.184` m, `0.479` m/s, `7.09` deg, `9.90` deg, `3.75` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.218` m, `0.240` m, `0.183` m, `0.475` m/s, `7.77` deg, `11.30` deg, `3.91` deg
- heldout/post_stall: samples `1420`, throws `10`
  - baseline dx/dy/alt/sink/roll/pitch/yaw MAE: `0.423` m, `0.766` m, `0.277` m, `0.412` m/s, `25.75` deg, `12.03` deg, `9.05` deg
  - candidate dx/dy/alt/sink/roll/pitch/yaw MAE: `0.622` m, `0.838` m, `0.339` m, `0.472` m/s, `34.57` deg, `18.64` deg, `8.87` deg

## Candidate Parameters

- baseline post-stall CL residual: `0`
- candidate post-stall CL residual: `0.0540889`
- baseline post-stall CD residual: `0`
- candidate post-stall CD residual: `-0.0764659`
- baseline post-stall Cm: `0`
- candidate post-stall Cm: `-0.124592`
- baseline post-stall Cmq: `0`
- candidate post-stall Cmq: `0`

## Regime Summary

- train/attached: count `130`, Cm mean `0.02802`, Cm MAE `0.02929`
- train/transition: count `538`, Cm mean `0.05084`, Cm MAE `0.06494`
- train/post_stall: count `2851`, Cm mean `-0.15531`, Cm MAE `0.16575`
- heldout/attached: count `73`, Cm mean `0.03199`, Cm MAE `0.03342`
- heldout/transition: count `267`, Cm mean `0.00843`, Cm MAE `0.09653`
- heldout/post_stall: count `1420`, Cm mean `-0.14198`, Cm MAE `0.15595`

## Interpretation

Accept the candidate only if held-out replay improves pitch/q behaviour without damaging x, altitude loss, or sink. Attached and transition residuals are diagnostic-only by default; accepted model changes should enter through the smoothly activated post-stall residual terms.
