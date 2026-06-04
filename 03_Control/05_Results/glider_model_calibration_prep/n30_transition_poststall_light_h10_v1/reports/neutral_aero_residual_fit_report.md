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
python 03_Control/02_Inner_Loop/run_fit_neutral_aero_residual_calibration.py --session-root 04_Flight_Test\05_Results\cal\n30 --run-label n30_transition_poststall_light_h10_v1 --heldout-count 10 --heldout-seed 606 --alignment-window-s 0.1 --derivative-window-s 0.04 --replay-dt-s 0.005 --ridge-lambda 0.001 --min-speed-m-s 1.5 --workers 8 --no-apply-attached-cm-bias --no-fit-post-stall-damping
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
