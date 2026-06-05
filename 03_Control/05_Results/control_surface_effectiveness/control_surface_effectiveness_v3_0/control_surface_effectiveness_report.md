# Real-Flight Control Surface Effectiveness Study v3.0

## 1. Purpose and Claim Boundary

The current neutral fitted model is frozen. Deflection ladder throws are used first as surface-effectiveness evidence and measured-command replay diagnostics. This is not broad aerodynamic SysID and does not claim accurate full 6-DoF lateral derivative identification.

- active neutral model: `neutral_dry_air_residual_calibrated_replay_n30_compact_coupled_v1`
- claim boundary: `control-surface effectiveness diagnostics and residual-calibrated replay alignment only; no broad aero SysID`
- promotion decision: `not_promoted`

## 2. Data Inventory

- inventoried throws: `214`
- kept for main effectiveness analysis: `174`
- filtered but retained in inventory: `40`
- train launches: `144`
- held-out launches: `30`

## 3. Filtering Rules

Main analysis requires valid state samples, a matching nonzero 20 percent command-lattice schedule, sufficient response time after command onset, relaxed replay-start velocity bounds, and no floor/wall/contact before the response window. Deep post-stall, launch asymmetry, rate outliers, and filtered throws remain reported rather than deleted.

- all throws: `214` total, `174` kept, `40` filtered; reason counts `{"command_timing_mismatch": 2, "invalid_or_cancelled_throw": 34, "missing_command_schedule": 34, "missing_or_short_state_samples": 34, "nonfinite_launch_velocity": 34, "outside_relaxed_replay_start_velocity_gate": 4, "response_window_too_short": 34}`

## 4. Frozen-Model Replay Setup

Each usable throw is replayed from its measured launch state using the active neutral calibrated model, logged command schedule, nominal command-onset delay, and actuator lag from the throw manifest.

- successful replays: `174` / `174`
- replay dx MAE: `0.251621292` m
- replay dy MAE: `0.5010647582` m
- replay altitude-loss MAE: `0.2276364641` m

## 5. Aileron Effectiveness

- `p_impulse_rad` at |cmd| `0.2`: real antisym `-0.1210443046`, frozen replay antisym `0.1505835379`, symmetric `-0.07303182681`
- `p_impulse_rad` at |cmd| `0.4`: real antisym `-0.2205070193`, frozen replay antisym `0.2555871341`, symmetric `-0.1208255662`
- `p_impulse_rad` at |cmd| `0.6`: real antisym `-0.2625987419`, frozen replay antisym `0.4490682297`, symmetric `-0.2407683994`
- `p_impulse_rad` at |cmd| `0.8`: real antisym `-0.3616444479`, frozen replay antisym `0.5715223463`, symmetric `-0.2237233768`
- `p_impulse_rad` at |cmd| `1`: real antisym `-0.4255499444`, frozen replay antisym `0.4802313681`, symmetric `-0.110179692`
- `peak_p_rad_s` at |cmd| `0.2`: real antisym `-0.3594961665`, frozen replay antisym `0.3171739093`, symmetric `-0.1930355039`
- `peak_p_rad_s` at |cmd| `0.4`: real antisym `-0.5748971197`, frozen replay antisym `0.4490394062`, symmetric `-0.2362419743`
- `peak_p_rad_s` at |cmd| `0.6`: real antisym `-0.5340032405`, frozen replay antisym `0.9708702694`, symmetric `-0.5629066085`
- `peak_p_rad_s` at |cmd| `0.8`: real antisym `-0.9441721259`, frozen replay antisym `1.313652468`, symmetric `-0.6744709449`
- `peak_p_rad_s` at |cmd| `1`: real antisym `-1.220090835`, frozen replay antisym `1.116029366`, symmetric `-0.2853769045`
- `phi_change_deg` at |cmd| `0.2`: real antisym `-4.978640333`, frozen replay antisym `9.813273846`, symmetric `-7.779638421`
- `phi_change_deg` at |cmd| `0.4`: real antisym `-12.7797718`, frozen replay antisym `15.42004346`, symmetric `-10.48163817`
- `phi_change_deg` at |cmd| `0.6`: real antisym `-17.26749192`, frozen replay antisym `25.60203648`, symmetric `-12.57800562`
- `phi_change_deg` at |cmd| `0.8`: real antisym `-21.46354104`, frozen replay antisym `28.23790525`, symmetric `-10.47514676`
- `phi_change_deg` at |cmd| `1`: real antisym `-23.93518186`, frozen replay antisym `24.56955055`, symmetric `-9.27870387`

## 6. Elevator Effectiveness

- `peak_q_rad_s` at |cmd| `0.2`: real antisym `-0.03035664165`, frozen replay antisym `0.6349204072`, symmetric `-1.794393169`
- `peak_q_rad_s` at |cmd| `0.4`: real antisym `0.1238087593`, frozen replay antisym `0.7109504954`, symmetric `-1.795648329`
- `peak_q_rad_s` at |cmd| `0.6`: real antisym `0.5185255419`, frozen replay antisym `1.065488168`, symmetric `-1.924735809`
- `peak_q_rad_s` at |cmd| `0.8`: real antisym `0.7856975071`, frozen replay antisym `2.045929059`, symmetric `-2.207331075`
- `peak_q_rad_s` at |cmd| `1`: real antisym `0.852705801`, frozen replay antisym `2.946757675`, symmetric `-2.184567145`
- `q_impulse_rad` at |cmd| `0.2`: real antisym `0.06495496952`, frozen replay antisym `0.2564347999`, symmetric `-0.6759196646`
- `q_impulse_rad` at |cmd| `0.4`: real antisym `0.1012180106`, frozen replay antisym `0.2677125986`, symmetric `-0.6678374679`
- `q_impulse_rad` at |cmd| `0.6`: real antisym `0.2910953912`, frozen replay antisym `0.4702313247`, symmetric `-0.7174272371`
- `q_impulse_rad` at |cmd| `0.8`: real antisym `0.3722199138`, frozen replay antisym `0.6875712462`, symmetric `-0.8612466801`
- `q_impulse_rad` at |cmd| `1`: real antisym `0.3734546679`, frozen replay antisym `0.850811136`, symmetric `-0.7616556321`
- `theta_change_deg` at |cmd| `0.2`: real antisym `3.064137532`, frozen replay antisym `12.35063522`, symmetric `-14.58694289`
- `theta_change_deg` at |cmd| `0.4`: real antisym `7.467085683`, frozen replay antisym `18.20159196`, symmetric `-14.61867972`
- `theta_change_deg` at |cmd| `0.6`: real antisym `14.74324526`, frozen replay antisym `25.62054167`, symmetric `-14.43100039`
- `theta_change_deg` at |cmd| `0.8`: real antisym `21.66625753`, frozen replay antisym `39.60053443`, symmetric `-19.6709983`
- `theta_change_deg` at |cmd| `1`: real antisym `23.77823652`, frozen replay antisym `52.19344001`, symmetric `-20.43839507`

## 7. Rudder Effectiveness

- `peak_r_rad_s` at |cmd| `0.2`: real antisym `0.0751527378`, frozen replay antisym `0.2055407106`, symmetric `0.2855270265`
- `peak_r_rad_s` at |cmd| `0.4`: real antisym `0.2858049148`, frozen replay antisym `0.5541984914`, symmetric `0.3242412438`
- `peak_r_rad_s` at |cmd| `0.6`: real antisym `0.3224312498`, frozen replay antisym `0.8872827261`, symmetric `0.3591352943`
- `peak_r_rad_s` at |cmd| `0.8`: real antisym `0.6973117494`, frozen replay antisym `1.153830789`, symmetric `0.1871974089`
- `peak_r_rad_s` at |cmd| `1`: real antisym `0.5984712171`, frozen replay antisym `1.394240526`, symmetric `0.3827552196`
- `psi_change_deg` at |cmd| `0.2`: real antisym `1.220462177`, frozen replay antisym `6.125477371`, symmetric `10.75778676`
- `psi_change_deg` at |cmd| `0.4`: real antisym `4.856261679`, frozen replay antisym `11.60153428`, symmetric `6.424287793`
- `psi_change_deg` at |cmd| `0.6`: real antisym `7.001545713`, frozen replay antisym `17.43957838`, symmetric `5.594880747`
- `psi_change_deg` at |cmd| `0.8`: real antisym `10.61294899`, frozen replay antisym `21.27665213`, symmetric `6.932046819`
- `psi_change_deg` at |cmd| `1`: real antisym `11.17866776`, frozen replay antisym `25.30250505`, symmetric `6.719284376`
- `r_impulse_rad` at |cmd| `0.2`: real antisym `0.008366203487`, frozen replay antisym `0.08723271807`, symmetric `0.1668012808`
- `r_impulse_rad` at |cmd| `0.4`: real antisym `0.0724427302`, frozen replay antisym `0.2070794792`, symmetric `0.1557162515`
- `r_impulse_rad` at |cmd| `0.6`: real antisym `0.1142859051`, frozen replay antisym `0.3256367725`, symmetric `0.1583579135`
- `r_impulse_rad` at |cmd| `0.8`: real antisym `0.1892594758`, frozen replay antisym `0.3926107903`, symmetric `0.1753381778`
- `r_impulse_rad` at |cmd| `1`: real antisym `0.1857516084`, frozen replay antisym `0.4572351387`, symmetric `0.2223284655`

## 8. Cross-Coupling Observations

Aileron yaw response and rudder roll response are reported as diagnostic coupling evidence. They are not promoted as lateral transition aerodynamic derivatives by this study.

## 9. Symmetric Launch/Trim Contamination

Symmetric response is separated from antisymmetric response. Large symmetric terms are interpreted as launch, trim, hardware, or model-mismatch contamination rather than hidden inside a surface effectiveness scale.

- aileron: mean absolute primary symmetric response `0.3904063872`
- elevator: mean absolute primary symmetric response `1.981335105`
- rudder: mean absolute primary symmetric response `0.3077712386`

## 10. Optional Surface-Only Fit Result

- `S0_frozen_neutral`: `evaluated_frozen_active_neutral_model`, promoted `False`
- `S1_surface_effectiveness_scales`: `diagnostic_metric_space_estimate_not_promoted`, promoted `False`
- `S2_scales_plus_neutral_biases`: `not_run_neutral_bias_not_fit_from_symmetric_contamination`, promoted `False`
- S1 held-out metric diagnostics improved `37` / `75` rows, but remain not promoted.

## 11. Promotion Decision

No model parameter is promoted by this analysis. A surface-only update would require held-out deflection improvement, neutral replay preservation, interpretable signs/magnitudes, and closed-loop smoke evidence.

## 12. Limitations

- Launch-condition contamination remains visible in the symmetric response.
- Deflection data are sustained pulse-ladder throws, not a broad aero excitation design.
- Optional S1/S2 rows are diagnostic summaries, not checked-in plant changes.
- R5/R7/R8/R10/R11 semantics are unchanged.

## 13. Reproducibility Commands

```powershell
python 03_Control/02_Inner_Loop/run_control_surface_effectiveness_study.py
pytest 03_Control/tests/test_control_surface_effectiveness_study.py
pytest 03_Control/tests/test_neutral_aero_residual_sysid_contract.py
```
