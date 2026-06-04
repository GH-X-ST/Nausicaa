# Neutral Aero Residual Regime Fit

This run uses only neutral open-loop real throws. It estimates force/moment residuals from Vicon state trajectories, fits regime-split pitch-moment coefficients, then validates the candidate by held-out dry-air replay.

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
python 03_Control/02_Inner_Loop/run_fit_neutral_aero_residual_calibration.py --session-root 04_Flight_Test\05_Results\cal\n30 --run-label n30_30throw_aero_residual_v1 --heldout-count 6 --heldout-seed 606 --alignment-window-s 0.1 --derivative-window-s 0.04 --replay-dt-s 0.005 --ridge-lambda 0.001 --min-speed-m-s 1.5 --workers 8 --no-apply-attached-cm-bias --no-fit-post-stall-damping
```

## Coefficient Fit

- fit status: `ok`
- sample count: `4247`
- used sample count: `4241`
- fit MAE in Cm: `0.08649`
- attached Cm residual: `0.0287722`
- transition Cm residual: `0.109507`
- post-stall Cm residual: `-0.118917`
- post-stall Cmq residual: `0`

## Replay Validation

- baseline train pitch MAE: `17.638` deg
- candidate train pitch MAE: `16.156` deg
- baseline held-out pitch MAE: `15.825` deg
- candidate held-out pitch MAE: `18.506` deg
- baseline held-out altitude-loss MAE: `0.4567` m
- candidate held-out altitude-loss MAE: `0.6984` m
- baseline held-out dx MAE: `0.9731` m
- candidate held-out dx MAE: `1.3822` m

## Candidate Parameters

- baseline post-stall Cm: `0`
- candidate post-stall Cm: `-0.118917`
- baseline post-stall Cmq: `0`
- candidate post-stall Cmq: `0`

## Regime Summary

- train/attached: count `148`, Cm mean `0.02642`, Cm MAE `0.02773`
- train/transition: count `626`, Cm mean `0.05083`, Cm MAE `0.06370`
- train/post_stall: count `3473`, Cm mean `-0.15090`, Cm MAE `0.16169`
- heldout/attached: count `55`, Cm mean `0.03759`, Cm MAE `0.03898`
- heldout/transition: count `179`, Cm mean `-0.01238`, Cm MAE `0.11638`
- heldout/post_stall: count `798`, Cm mean `-0.15078`, Cm MAE `0.16598`

## Interpretation

Accept the candidate only if held-out replay improves pitch/q behaviour without damaging x, altitude loss, or sink. If coefficient residuals look clean but replay worsens, this stage is diagnostic only and the model still needs a richer state such as separation lag/dynamic stall.
