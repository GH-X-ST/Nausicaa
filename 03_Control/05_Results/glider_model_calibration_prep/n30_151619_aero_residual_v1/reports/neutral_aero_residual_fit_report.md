# Neutral Aero Residual Regime Fit

This run uses only neutral open-loop real throws. It estimates force/moment residuals from Vicon state trajectories, fits regime-split pitch-moment coefficients, then validates the candidate by held-out dry-air replay.

## Rerun Recipe

- source session root: `04_Flight_Test\05_Results\cal\n30\20260604_151619`
- alignment window: `0.100` s
- derivative window: `0.040` s
- replay dt: `0.0050` s
- ridge lambda: `0.001`
- min speed: `1.50` m/s
- workers: `8`
- apply attached Cm bias: `False`

```powershell
python 03_Control/02_Inner_Loop/run_fit_neutral_aero_residual_calibration.py --session-root 04_Flight_Test\05_Results\cal\n30\20260604_151619 --run-label n30_151619_aero_residual_v1 --heldout-count 2 --heldout-seed 606 --alignment-window-s 0.1 --derivative-window-s 0.04 --replay-dt-s 0.005 --ridge-lambda 0.001 --min-speed-m-s 1.5 --workers 8 --no-apply-attached-cm-bias
```

## Coefficient Fit

- fit status: `ok`
- sample count: `1488`
- used sample count: `1482`
- fit MAE in Cm: `0.04983`
- attached Cm residual: `0.0301539`
- transition Cm residual: `0.0248784`
- post-stall Cm residual: `-0.0680859`
- post-stall Cmq residual: `5.204`

## Replay Validation

- baseline train pitch MAE: `16.517` deg
- candidate train pitch MAE: `22.950` deg
- baseline held-out pitch MAE: `23.455` deg
- candidate held-out pitch MAE: `29.023` deg
- baseline held-out altitude-loss MAE: `1.0659` m
- candidate held-out altitude-loss MAE: `1.2732` m
- baseline held-out dx MAE: `1.8390` m
- candidate held-out dx MAE: `2.3097` m

## Candidate Parameters

- baseline post-stall Cm: `0`
- candidate post-stall Cm: `-0.0680859`
- baseline post-stall Cmq: `0`
- candidate post-stall Cmq: `4`

## Regime Summary

- train/attached: count `72`, Cm mean `0.03028`, Cm MAE `0.03208`
- train/transition: count `205`, Cm mean `0.05282`, Cm MAE `0.05552`
- train/post_stall: count `1211`, Cm mean `-0.16305`, Cm MAE `0.16998`
- heldout/attached: count `20`, Cm mean `0.03480`, Cm MAE `0.03549`
- heldout/transition: count `84`, Cm mean `-0.01445`, Cm MAE `0.13091`
- heldout/post_stall: count `276`, Cm mean `-0.14551`, Cm MAE `0.17091`

## Interpretation

Accept the candidate only if held-out replay improves pitch/q behaviour without damaging x, altitude loss, or sink. If coefficient residuals look clean but replay worsens, this stage is diagnostic only and the model still needs a richer state such as separation lag/dynamic stall.
