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
- replay dx MAE: `0.2514345921` m
- replay dy MAE: `0.4848302572` m
- replay altitude-loss MAE: `0.2202974012` m

## 5. Candidate Replay Error Summary

The candidate comparison fits only launch-confidence-weighted residual surface aero/coupling derivatives. Surface-scale fitting is not part of the default fit because measured surface magnitudes are already used; scaling remains an optional legacy appendix only.

- successful candidate-family replays: `870` / `870`

- replay MAE comparison; lower is better:
`candidate | split | surface | dx | dy | altitude | phi | theta | psi | primary antisym`
- `C0_frozen_neutral | all | all | 0.2514345921 | 0.4848302572 | 0.2202974012 | 18.62355485 | 14.14598559 | 15.70785639 | 0.5482246838`
- `C0_frozen_neutral | all | aileron | 0.3777756604 | 0.5492673767 | 0.3277644207 | 25.45684136 | 10.31288765 | 22.49422645 | 0.170756809`
- `C0_frozen_neutral | all | elevator | 0.1757974962 | 0.405569816 | 0.179223145 | 15.29032825 | 25.40951725 | 10.91684148 | 1.030732967`
- `C0_frozen_neutral | all | rudder | 0.1972482375 | 0.4971760348 | 0.1513435815 | 14.94821023 | 6.975838602 | 13.51289116 | 0.4431842749`
- `C0_frozen_neutral | heldout | all | 0.2675387269 | 0.4169147553 | 0.1988436646 | 18.920798 | 15.5982777 | 19.22118409 | 0.5166743763`
- `C0_frozen_neutral | heldout | aileron | 0.3905845536 | 0.3820599909 | 0.3055433691 | 24.09346252 | 11.46736938 | 30.09453911 | 0.1848068848`
- `C0_frozen_neutral | heldout | elevator | 0.1725122949 | 0.3704089581 | 0.1251236699 | 18.69869195 | 26.22830299 | 12.96357166 | 0.793614837`
- `C0_frozen_neutral | heldout | rudder | 0.2395193321 | 0.498275317 | 0.1658639549 | 13.97023953 | 9.099160718 | 14.6054415 | 0.5716014072`
- `C1_primary_moment_derivatives | all | all | 0.2573691983 | 0.4728253528 | 0.2115176202 | 17.76223078 | 12.43212241 | 12.75802275 | 0.2839926702`
- `C1_primary_moment_derivatives | all | aileron | 0.3764298926 | 0.5458803669 | 0.3319113355 | 23.07618194 | 9.629541515 | 19.83635294 | 0.2859409949`
- `C1_primary_moment_derivatives | all | elevator | 0.1654026846 | 0.4111593462 | 0.1469108698 | 13.86154493 | 20.73542407 | 9.339249411 | 0.3506615579`
- `C1_primary_moment_derivatives | all | rudder | 0.2266366176 | 0.4591135691 | 0.1525409922 | 16.19009241 | 7.122882371 | 8.917481347 | 0.2153754577`
- `C1_primary_moment_derivatives | heldout | all | 0.2655753104 | 0.4028789231 | 0.1834969157 | 17.39515811 | 13.36095634 | 14.64113151 | 0.2850220724`
- `C1_primary_moment_derivatives | heldout | aileron | 0.3826210308 | 0.3833671842 | 0.3065437944 | 22.16052622 | 12.12850277 | 27.25250838 | 0.2787030309`
- `C1_primary_moment_derivatives | heldout | elevator | 0.1397784638 | 0.3772057634 | 0.08785859034 | 14.37436082 | 19.28190516 | 7.684142728 | 0.2159291286`
- `C1_primary_moment_derivatives | heldout | rudder | 0.2743264368 | 0.4480638218 | 0.1560883623 | 15.65058728 | 8.672461085 | 8.986743411 | 0.3604340579`
- `C2_c1_plus_aileron_adverse_yaw | all | all | 0.2340307302 | 0.5251921038 | 0.2299574399 | 16.91578191 | 14.12252257 | 10.36444682 | 0.2440079754`
- `C2_c1_plus_aileron_adverse_yaw | all | aileron | 0.3076011901 | 0.7003179038 | 0.3862931765 | 20.5798751 | 14.61478946 | 12.7773324 | 0.1659869107`
- `C2_c1_plus_aileron_adverse_yaw | all | elevator | 0.1654026846 | 0.4111593462 | 0.1469108698 | 13.86154493 | 20.73542407 | 9.339249411 | 0.3506615579`
- `C2_c1_plus_aileron_adverse_yaw | all | rudder | 0.2266366176 | 0.4591135691 | 0.1525409922 | 16.19009241 | 7.122882371 | 8.917481347 | 0.2153754577`
- `C2_c1_plus_aileron_adverse_yaw | heldout | all | 0.2431454828 | 0.5226012453 | 0.1954337908 | 16.46772378 | 14.39672827 | 10.98422201 | 0.2622295743`
- `C2_c1_plus_aileron_adverse_yaw | heldout | aileron | 0.3153315478 | 0.7425341505 | 0.3423544197 | 19.37822324 | 15.23581855 | 16.28177988 | 0.2103255366`
- `C2_c1_plus_aileron_adverse_yaw | heldout | elevator | 0.1397784638 | 0.3772057634 | 0.08785859034 | 14.37436082 | 19.28190516 | 7.684142728 | 0.2159291286`
- `C2_c1_plus_aileron_adverse_yaw | heldout | rudder | 0.2743264368 | 0.4480638218 | 0.1560883623 | 15.65058728 | 8.672461085 | 8.986743411 | 0.3604340579`
- `C3_c1_plus_rudder_roll | all | all | 0.2574425472 | 0.4725048911 | 0.2113328347 | 17.84115959 | 12.45390037 | 12.74678499 | 0.2852319912`
- `C3_c1_plus_rudder_roll | all | aileron | 0.3764298926 | 0.5458803669 | 0.3319113355 | 23.07618194 | 9.629541515 | 19.83635294 | 0.2859409949`
- `C3_c1_plus_rudder_roll | all | elevator | 0.1654026846 | 0.4111593462 | 0.1469108698 | 13.86154493 | 20.73542407 | 9.339249411 | 0.3506615579`
- `C3_c1_plus_rudder_roll | all | rudder | 0.2268566643 | 0.4581521839 | 0.1519866357 | 16.42687885 | 7.188216248 | 8.883768085 | 0.2190934208`
- `C3_c1_plus_rudder_roll | heldout | all | 0.2656354494 | 0.401943469 | 0.1831694955 | 17.41212169 | 13.37546645 | 14.67131262 | 0.2845376119`
- `C3_c1_plus_rudder_roll | heldout | aileron | 0.3826210308 | 0.3833671842 | 0.3065437944 | 22.16052622 | 12.12850277 | 27.25250838 | 0.2787030309`
- `C3_c1_plus_rudder_roll | heldout | elevator | 0.1397784638 | 0.3772057634 | 0.08785859034 | 14.37436082 | 19.28190516 | 7.684142728 | 0.2159291286`
- `C3_c1_plus_rudder_roll | heldout | rudder | 0.2745068536 | 0.4452574595 | 0.1551061017 | 15.70147801 | 8.715991423 | 9.077286741 | 0.3589806763`
- `C4_c1_plus_surface_side_force | all | all | 0.2605122002 | 0.4659017836 | 0.2011185608 | 17.93456303 | 12.1686904 | 13.48222245 | 0.2686635899`
- `C4_c1_plus_surface_side_force | all | aileron | 0.3816793375 | 0.5531536448 | 0.3020070436 | 23.97290078 | 8.751928922 | 22.95199868 | 0.2784646893`
- `C4_c1_plus_surface_side_force | all | elevator | 0.1654026846 | 0.4111593462 | 0.1469108698 | 13.86154493 | 20.73542407 | 9.339249411 | 0.3506615579`
- `C4_c1_plus_surface_side_force | all | rudder | 0.2307256707 | 0.4309441824 | 0.1517636971 | 15.79490965 | 7.225330182 | 7.920716672 | 0.1768645225`
- `C4_c1_plus_surface_side_force | heldout | all | 0.2707336786 | 0.3734292208 | 0.1752459911 | 17.69782463 | 13.22527186 | 15.5230233 | 0.2694957713`
- `C4_c1_plus_surface_side_force | heldout | aileron | 0.393671352 | 0.3258758115 | 0.2872497906 | 23.18014115 | 11.58284816 | 30.98973977 | 0.2652312815`
- `C4_c1_plus_surface_side_force | heldout | elevator | 0.1397784638 | 0.3772057634 | 0.08785859034 | 14.37436082 | 19.28190516 | 7.684142728 | 0.2159291286`
- `C4_c1_plus_surface_side_force | heldout | rudder | 0.2787512198 | 0.4172060875 | 0.1506295924 | 15.53897191 | 8.81106227 | 7.89518742 | 0.3273269038`
- `C5_c2_plus_surface_side_force | all | all | 0.2446886567 | 0.4905698267 | 0.2189972492 | 17.22025814 | 13.83405932 | 9.628445316 | 0.2325258866`
- `C5_c2_plus_surface_side_force | all | aileron | 0.3350132941 | 0.6259034666 | 0.354734023 | 21.8663067 | 13.6633559 | 11.58662206 | 0.1700515793`
- `C5_c2_plus_surface_side_force | all | elevator | 0.1654026846 | 0.4111593462 | 0.1469108698 | 13.86154493 | 20.73542407 | 9.339249411 | 0.3506615579`
- `C5_c2_plus_surface_side_force | all | rudder | 0.2307256707 | 0.4309441824 | 0.1517636971 | 15.79490965 | 7.225330182 | 7.920716672 | 0.1768645225`
- `C5_c2_plus_surface_side_force | heldout | all | 0.2522202557 | 0.462863049 | 0.1872271061 | 16.74357763 | 14.20556275 | 10.51280006 | 0.2494876854`
- `C5_c2_plus_surface_side_force | heldout | aileron | 0.3381310834 | 0.5941772961 | 0.3231931357 | 20.31740015 | 14.52372081 | 15.95907003 | 0.2052070239`
- `C5_c2_plus_surface_side_force | heldout | elevator | 0.1397784638 | 0.3772057634 | 0.08785859034 | 14.37436082 | 19.28190516 | 7.684142728 | 0.2159291286`
- `C5_c2_plus_surface_side_force | heldout | rudder | 0.2787512198 | 0.4172060875 | 0.1506295924 | 15.53897191 | 8.81106227 | 7.89518742 | 0.3273269038`

## 6. Launch-Confidence Diagnostic

Launch confidence is a diagnostic weight and grouping variable, not a new acceptance gate. It reuses the neutral SysID lateral-contamination strategy with reference `phi0=psi0=v0=p0=r0=0`, so the study can test whether real-vs-replay mismatch is launch-condition driven.

- all successful replays: `174` total, `139` high-confidence, `35` medium-confidence, `0` low-confidence; mean confidence weight `0.8246298746`, mean lateral-contamination score `0.3490437321`
- primary antisymmetric residual check; lower is better, negative delta means the confidence subset reduced mismatch:
- aileron: all `0.170756809`, high-confidence `0.2536082227` (delta `0.08285141368`), weighted `0.1756967213` (delta `0.004939912343`)
- elevator: all `1.030732967`, high-confidence `1.025348524` (delta `-0.005384443176`), weighted `1.017842185` (delta `-0.01289078237`)
- rudder: all `0.4431842749`, high-confidence `0.4293002134` (delta `-0.01388406144`), weighted `0.4420018205` (delta `-0.001182454419`)

## 7. Aileron Effectiveness

- `p_impulse_rad` at |cmd| `0.2`: real antisym `-0.1210443046`, frozen replay antisym `-0.1557525924`, symmetric `-0.07303182681`
- `p_impulse_rad` at |cmd| `0.4`: real antisym `-0.2205070193`, frozen replay antisym `-0.3238383061`, symmetric `-0.1208255662`
- `p_impulse_rad` at |cmd| `0.6`: real antisym `-0.2625987419`, frozen replay antisym `-0.5133613713`, symmetric `-0.2407683994`
- `p_impulse_rad` at |cmd| `0.8`: real antisym `-0.3616444479`, frozen replay antisym `-0.4587645602`, symmetric `-0.2237233768`
- `p_impulse_rad` at |cmd| `1`: real antisym `-0.4255499444`, frozen replay antisym `-0.5211829091`, symmetric `-0.110179692`
- `peak_p_rad_s` at |cmd| `0.2`: real antisym `-0.3594961665`, frozen replay antisym `-0.2462101975`, symmetric `-0.1930355039`
- `peak_p_rad_s` at |cmd| `0.4`: real antisym `-0.5748971197`, frozen replay antisym `-0.6397697748`, symmetric `-0.2362419743`
- `peak_p_rad_s` at |cmd| `0.6`: real antisym `-0.5340032405`, frozen replay antisym `-1.131940715`, symmetric `-0.5629066085`
- `peak_p_rad_s` at |cmd| `0.8`: real antisym `-0.9441721259`, frozen replay antisym `-0.9178158059`, symmetric `-0.6744709449`
- `peak_p_rad_s` at |cmd| `1`: real antisym `-1.220090835`, frozen replay antisym `-1.271422462`, symmetric `-0.2853769045`
- `phi_change_deg` at |cmd| `0.2`: real antisym `-4.978640333`, frozen replay antisym `-7.704175762`, symmetric `-7.779638421`
- `phi_change_deg` at |cmd| `0.4`: real antisym `-12.7797718`, frozen replay antisym `-16.93710236`, symmetric `-10.48163817`
- `phi_change_deg` at |cmd| `0.6`: real antisym `-17.26749192`, frozen replay antisym `-28.12835956`, symmetric `-12.57800562`
- `phi_change_deg` at |cmd| `0.8`: real antisym `-21.46354104`, frozen replay antisym `-28.18708176`, symmetric `-10.47514676`
- `phi_change_deg` at |cmd| `1`: real antisym `-23.93518186`, frozen replay antisym `-27.15055966`, symmetric `-9.27870387`

## 8. Elevator Effectiveness

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

## 9. Rudder Effectiveness

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

## 10. Cross-Coupling Observations

Aileron yaw response and rudder roll response are reported as diagnostic coupling evidence. They are not promoted as lateral transition aerodynamic derivatives by this study.

## 11. Symmetric Launch/Trim Contamination

Symmetric response is separated from antisymmetric response. Large symmetric terms are interpreted as launch, trim, hardware, or model-mismatch contamination rather than hidden inside a surface effectiveness scale.

- aileron: mean absolute primary symmetric response `0.3904063872`
- elevator: mean absolute primary symmetric response `1.981335105`
- rudder: mean absolute primary symmetric response `0.3077712386`

## 12. Optional Surface/Aero Fit Result

- `S0_frozen_neutral`: `evaluated_frozen_active_neutral_model`, promoted `False`
- `S1_surface_effectiveness_scales`: `not_run_disabled_by_cli`, promoted `False`
- `S2_scales_plus_neutral_biases`: `not_run_disabled_by_cli`, promoted `False`
- `D0_launch_confidence_weighted_derivative_fit_basis`: `diagnostic_derivative_level_fit_not_promoted`, promoted `False`
- S1/S2 surface-scale diagnostics are disabled by default because measured surface magnitudes are used.

- The derivative diagnostic fits residual control force/moment coefficients from measured acceleration with launch-confidence weighting; it is not replay-promoted.
- `CY_delta_a_residual`: coeff `0.4798183808`, held-out baseline `0.2512200618`, candidate `0.226674254`, improved `True`
- `Cl_delta_a_residual`: coeff `0.05628318768`, held-out baseline `0.01936537792`, candidate `0.01595728473`, improved `True`
- `Cn_delta_a_residual`: coeff `-0.08210197202`, held-out baseline `0.01912805222`, candidate `0.01377962783`, improved `True`
- `Cm_delta_e_residual`: coeff `-0.115511593`, held-out baseline `0.09140446186`, candidate `0.0845325132`, improved `True`
- `CY_delta_r_residual`: coeff `-0.1708085654`, held-out baseline `0.2371454031`, candidate `0.2394389883`, improved `False`
- `Cn_delta_r_residual`: coeff `-0.01043556406`, held-out baseline `0.01338416377`, candidate `0.01290536735`, improved `True`
- `Cl_delta_r_residual`: coeff `-0.004294737192`, held-out baseline `0.01286848089`, candidate `0.01276270234`, improved `True`

## 13. Promotion Decision

No model parameter is promoted by this analysis. A surface-only update would require held-out deflection improvement, neutral replay preservation, interpretable signs/magnitudes, and closed-loop smoke evidence.

## 14. Limitations

- Launch-condition contamination remains visible in the symmetric response.
- Deflection data are sustained pulse-ladder throws, not a broad aero excitation design.
- Candidate derivative rows are diagnostic summaries, not checked-in plant changes.
- S1/S2 surface-scale rows are disabled by default because measured surface magnitudes are already used.
- R5/R7/R8/R10/R11 semantics are unchanged.

## 15. Reproducibility Commands

```powershell
python 03_Control/02_Inner_Loop/run_control_surface_effectiveness_study.py
pytest 03_Control/tests/test_control_surface_effectiveness_study.py
pytest 03_Control/tests/test_neutral_aero_residual_sysid_contract.py
```
