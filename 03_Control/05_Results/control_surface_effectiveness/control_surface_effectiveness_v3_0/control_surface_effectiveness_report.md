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

The candidate comparison fits only launch-confidence-weighted residual surface aero/coupling derivatives. Surface-scale fitting is not part of the default fit because measured surface magnitudes are already used; scaling remains an optional legacy appendix only. C6-C8 add a diagnostic alpha-regime schedule using normal, transition, and post-stall bins; rudder post-stall shares the transition coefficient because the kept data have no held-out rudder post-stall support.

- successful candidate-family replays: `1392` / `1392`

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
- `C6_alpha_regime_primary_derivatives | all | all | 0.2718361435 | 0.4605037278 | 0.235692287 | 19.52814489 | 8.996936201 | 11.71363289 | 0.3220972817`
- `C6_alpha_regime_primary_derivatives | all | aileron | 0.3760232314 | 0.5442848505 | 0.334622431 | 26.03556166 | 9.558351287 | 17.97446225 | 0.3831466476`
- `C6_alpha_regime_primary_derivatives | all | elevator | 0.1809803796 | 0.4127370252 | 0.2142059048 | 14.8531966 | 10.38132744 | 9.417859973 | 0.3840750098`
- `C6_alpha_regime_primary_derivatives | all | rudder | 0.2551420117 | 0.4222212417 | 0.1561723783 | 17.502877 | 7.065319119 | 7.60104883 | 0.1990701877`
- `C6_alpha_regime_primary_derivatives | heldout | all | 0.2870565194 | 0.3769748797 | 0.2012595537 | 21.13629028 | 10.96523105 | 14.67242785 | 0.3987514016`
- `C6_alpha_regime_primary_derivatives | heldout | aileron | 0.3773442575 | 0.3908265365 | 0.3087317665 | 26.89736596 | 12.03690164 | 23.45956768 | 0.3924613899`
- `C6_alpha_regime_primary_derivatives | heldout | elevator | 0.1766683298 | 0.3754674184 | 0.1508061703 | 19.39145755 | 12.8401523 | 12.7762494 | 0.2253603569`
- `C6_alpha_regime_primary_derivatives | heldout | rudder | 0.3071569707 | 0.3646306841 | 0.1442407244 | 17.12004734 | 8.018639192 | 7.781466469 | 0.5784324581`
- `C7_c6_plus_alpha_regime_aileron_yaw | all | all | 0.2710663672 | 0.4831082538 | 0.244027824 | 17.41180644 | 9.746892035 | 8.845728095 | 0.2677747287`
- `C7_c6_plus_alpha_regime_aileron_yaw | all | aileron | 0.3737530438 | 0.610949046 | 0.3592052012 | 19.79415675 | 11.77008544 | 9.516573524 | 0.2201789888`
- `C7_c6_plus_alpha_regime_aileron_yaw | all | elevator | 0.1809803796 | 0.4127370252 | 0.2142059048 | 14.8531966 | 10.38132744 | 9.417859973 | 0.3840750098`
- `C7_c6_plus_alpha_regime_aileron_yaw | all | rudder | 0.2551420117 | 0.4222212417 | 0.1561723783 | 17.502877 | 7.065319119 | 7.60104883 | 0.1990701877`
- `C7_c6_plus_alpha_regime_aileron_yaw | heldout | all | 0.2872789031 | 0.4672308383 | 0.2039456299 | 18.63210917 | 11.41969764 | 9.372368417 | 0.3343856466`
- `C7_c6_plus_alpha_regime_aileron_yaw | heldout | aileron | 0.3780114087 | 0.6615944124 | 0.3167899951 | 19.38482261 | 13.40030143 | 7.559389384 | 0.1993641247`
- `C7_c6_plus_alpha_regime_aileron_yaw | heldout | elevator | 0.1766683298 | 0.3754674184 | 0.1508061703 | 19.39145755 | 12.8401523 | 12.7762494 | 0.2253603569`
- `C7_c6_plus_alpha_regime_aileron_yaw | heldout | rudder | 0.3071569707 | 0.3646306841 | 0.1442407244 | 17.12004734 | 8.018639192 | 7.781466469 | 0.5784324581`
- `C8_c7_plus_alpha_regime_aileron_side_force | all | all | 0.2570507017 | 0.523109757 | 0.2640680757 | 16.6779142 | 10.13884526 | 9.332977619 | 0.2634107389`
- `C8_c7_plus_alpha_regime_aileron_side_force | all | aileron | 0.332418708 | 0.7289195808 | 0.4183069604 | 17.62979659 | 12.92601528 | 10.9535467 | 0.2070870192`
- `C8_c7_plus_alpha_regime_aileron_side_force | all | elevator | 0.1809803796 | 0.4127370252 | 0.2142059048 | 14.8531966 | 10.38132744 | 9.417859973 | 0.3840750098`
- `C8_c7_plus_alpha_regime_aileron_side_force | all | rudder | 0.2551420117 | 0.4222212417 | 0.1561723783 | 17.502877 | 7.065319119 | 7.60104883 | 0.1990701877`
- `C8_c7_plus_alpha_regime_aileron_side_force | heldout | all | 0.2716410681 | 0.44831933 | 0.2336448567 | 18.02898782 | 11.55601095 | 10.67094442 | 0.3363356842`
- `C8_c7_plus_alpha_regime_aileron_side_force | heldout | aileron | 0.3310979038 | 0.6048598876 | 0.4058876755 | 17.57545857 | 13.80924136 | 11.45511739 | 0.2052142377`
- `C8_c7_plus_alpha_regime_aileron_side_force | heldout | elevator | 0.1766683298 | 0.3754674184 | 0.1508061703 | 19.39145755 | 12.8401523 | 12.7762494 | 0.2253603569`
- `C8_c7_plus_alpha_regime_aileron_side_force | heldout | rudder | 0.3071569707 | 0.3646306841 | 0.1442407244 | 17.12004734 | 8.018639192 | 7.781466469 | 0.5784324581`

## 6. Alpha-Regime Command-Ladder Replay Error

Replay error is also reported as an explicit candidate/surface/alpha-regime/20 percent command ladder. Regime is assigned from measured response-window `actual_max_abs_alpha_deg`: normal `<12 deg`, transition `12-22 deg`, and post-stall `>=22 deg`. Empty cells are retained in `control_surface_regime_ladder_error_summary.csv` with `replay_count=0` so missing support is visible.

- held-out non-empty cells; lower is better:
`candidate | surface | regime | |cmd| | n | dx | dy | altitude | phi | theta | psi | primary`
- `C0_frozen_neutral | aileron | transition | 0.2 | 2 | 0.4504285959 | 0.2530196281 | 0.2451856179 | 19.76557821 | 6.103107888 | 7.720275058 | 0.1661385645`
- `C0_frozen_neutral | aileron | transition | 0.4 | 2 | 0.3759545656 | 0.5306732028 | 0.4728628765 | 18.73200363 | 11.49486243 | 6.540730445 | 0.1738168885`
- `C0_frozen_neutral | aileron | transition | 0.6 | 1 | 0.8681745221 | 0.01671321325 | 0.9611264127 | 39.10448952 | 17.06961034 | 7.456566527 | `
- `C0_frozen_neutral | aileron | transition | 0.8 | 2 | 0.1166013734 | 0.1255892131 | 0.02820657031 | 24.04501034 | 7.753203696 | 39.72699223 | 0.1253302209`
- `C0_frozen_neutral | aileron | post_stall | 0.6 | 1 | 0.04340959713 | 0.417080028 | 0.1162668355 | 5.920577236 | 1.641215174 | 57.88166922 | `
- `C0_frozen_neutral | aileron | post_stall | 1 | 2 | 0.5541461733 | 0.7841212898 | 0.2427651567 | 35.41218707 | 22.63026016 | 63.81557994 | 0.4574340949`
- `C0_frozen_neutral | elevator | transition | 0.2 | 1 | 0.05023066691 | 0.189376071 | 0.04696263459 | 28.0634927 | 7.00036721 | 12.63791336 | `
- `C0_frozen_neutral | elevator | transition | 0.4 | 1 | 0.1964410202 | 0.05491394543 | 0.08067344268 | 16.85522325 | 17.17434574 | 0.2212808546 | `
- `C0_frozen_neutral | elevator | transition | 0.6 | 1 | 0.0212352141 | 0.5829752872 | 0.08026931405 | 12.92575893 | 1.608224141 | 4.347481836 | `
- `C0_frozen_neutral | elevator | transition | 0.8 | 1 | 0.1474302208 | 0.5716069823 | 0.2166642797 | 13.70965065 | 17.73033584 | 1.921632099 | `
- `C0_frozen_neutral | elevator | transition | 1 | 1 | 0.06619721783 | 0.2818219827 | 0.2309245087 | 46.59772689 | 41.07831105 | 55.27653368 | `
- `C0_frozen_neutral | elevator | post_stall | 0.2 | 1 | 0.07377152192 | 0.5704423204 | 0.03194167668 | 13.33987804 | 15.32069052 | 11.58579798 | `
- `C0_frozen_neutral | elevator | post_stall | 0.4 | 1 | 0.2782885137 | 0.189008803 | 0.1833255381 | 21.09478719 | 39.2451072 | 8.440578889 | `
- `C0_frozen_neutral | elevator | post_stall | 0.6 | 1 | 0.3494927809 | 0.3664829191 | 0.09931426417 | 16.19273398 | 35.68639132 | 14.18198174 | `
- `C0_frozen_neutral | elevator | post_stall | 0.8 | 1 | 0.3457062411 | 0.6387822363 | 0.2063742748 | 12.58653535 | 44.16940305 | 8.88784298 | `
- `C0_frozen_neutral | elevator | post_stall | 1 | 1 | 0.1963295512 | 0.2586790334 | 0.07478676522 | 5.621132523 | 43.26985385 | 12.13467312 | `
- `C0_frozen_neutral | rudder | transition | 0.2 | 2 | 0.377380812 | 0.1265475992 | 0.3734611931 | 16.96988245 | 9.324813503 | 7.770061952 | 0.8218151753`
- `C0_frozen_neutral | rudder | transition | 0.4 | 2 | 0.1352121599 | 0.3562545114 | 0.08019368549 | 19.52697663 | 12.17784025 | 6.426630687 | 0.5711831517`
- `C0_frozen_neutral | rudder | transition | 0.6 | 2 | 0.134123656 | 0.3831172802 | 0.02384417233 | 12.43261012 | 1.886094046 | 8.349837733 | 0.2906263145`
- `C0_frozen_neutral | rudder | transition | 0.8 | 2 | 0.2057581688 | 0.7373711728 | 0.1490897537 | 8.530993454 | 16.05666527 | 26.47619571 | 0.5931191604`
- `C0_frozen_neutral | rudder | transition | 1 | 2 | 0.3451218638 | 0.8880860216 | 0.20273097 | 12.39073498 | 6.050390518 | 24.0044814 | 0.5812632339`
- `C1_primary_moment_derivatives | aileron | transition | 0.2 | 2 | 0.4499052779 | 0.2385401049 | 0.2368853984 | 20.28372936 | 6.598700005 | 6.685003006 | 0.2948156863`
- `C1_primary_moment_derivatives | aileron | transition | 0.4 | 2 | 0.3738235596 | 0.5000816506 | 0.4664340154 | 11.41963098 | 10.74652327 | 5.702543456 | 0.3357760766`
- `C1_primary_moment_derivatives | aileron | transition | 0.6 | 1 | 0.8644254084 | 0.04103804474 | 0.935824806 | 28.24866505 | 15.18903256 | 9.92364156 | `
- `C1_primary_moment_derivatives | aileron | transition | 0.8 | 2 | 0.106013837 | 0.1095272163 | 0.02243142495 | 24.83714723 | 5.34580526 | 34.10127681 | 0.3311265956`
- `C1_primary_moment_derivatives | aileron | post_stall | 0.6 | 1 | 0.02258546157 | 0.4559191382 | 0.117964632 | 17.173813 | 5.515410421 | 51.58472659 | `
- `C1_primary_moment_derivatives | aileron | post_stall | 1 | 2 | 0.5398570444 | 0.8202083576 | 0.280073414 | 31.55088453 | 27.59926382 | 59.01953457 | 0.159993069`
- `C1_primary_moment_derivatives | elevator | transition | 0.2 | 1 | 0.03070998963 | 0.1938428766 | 0.04985253454 | 23.94031424 | 5.900158222 | 12.2812985 | `
- `C1_primary_moment_derivatives | elevator | transition | 0.4 | 1 | 0.1454193795 | 0.05190277819 | 0.07947425735 | 24.38553704 | 15.67538243 | 2.44273692 | `
- `C1_primary_moment_derivatives | elevator | transition | 0.6 | 1 | 0.03934116528 | 0.5815338217 | 0.01857658997 | 13.41742582 | 10.55319411 | 4.821640108 | `
- `C1_primary_moment_derivatives | elevator | transition | 0.8 | 1 | 0.1587742928 | 0.5680059618 | 0.04533238147 | 13.58822376 | 17.57895223 | 2.891770581 | `
- `C1_primary_moment_derivatives | elevator | transition | 1 | 1 | 0.1013597988 | 0.3510242772 | 0.2650384008 | 0.7193092427 | 3.759516568 | 5.008987048 | `
- `C1_primary_moment_derivatives | elevator | post_stall | 0.2 | 1 | 0.04773793034 | 0.5720180782 | 0.008904509661 | 13.21487975 | 11.33360967 | 11.16885113 | `
- `C1_primary_moment_derivatives | elevator | post_stall | 0.4 | 1 | 0.4124999572 | 0.1803021919 | 0.1398588682 | 20.18928767 | 31.18559279 | 5.930292536 | `
- `C1_primary_moment_derivatives | elevator | post_stall | 0.6 | 1 | 0.1982575792 | 0.3663681206 | 0.07227879954 | 17.3135541 | 26.82477811 | 12.3478575 | `
- `C1_primary_moment_derivatives | elevator | post_stall | 0.8 | 1 | 0.1960426855 | 0.6496815002 | 0.163377586 | 12.52227733 | 35.13387609 | 8.997510651 | `
- `C1_primary_moment_derivatives | elevator | post_stall | 1 | 1 | 0.06764185979 | 0.2573780277 | 0.03589197592 | 4.452799237 | 34.87399141 | 10.95048231 | `
- `C1_primary_moment_derivatives | rudder | transition | 0.2 | 2 | 0.3809946242 | 0.1102183267 | 0.3711026717 | 17.20091547 | 9.146012548 | 6.414536254 | 0.8472201067`
- `C1_primary_moment_derivatives | rudder | transition | 0.4 | 2 | 0.1366673066 | 0.3763509402 | 0.07934436928 | 20.30980673 | 12.24203892 | 7.047815213 | 0.4839540972`
- `C1_primary_moment_derivatives | rudder | transition | 0.6 | 2 | 0.1315641988 | 0.4105132777 | 0.02337024279 | 13.17564231 | 2.292803702 | 5.036221087 | 0.1644326789`
- `C1_primary_moment_derivatives | rudder | transition | 0.8 | 2 | 0.2585862069 | 0.6319623159 | 0.1407733397 | 11.25739586 | 15.00443122 | 16.81721466 | 0.2799716288`
- `C1_primary_moment_derivatives | rudder | transition | 1 | 2 | 0.4638198473 | 0.7112742487 | 0.1658511882 | 16.30917602 | 4.677019034 | 9.617929843 | 0.02659177764`
- `C2_c1_plus_aileron_adverse_yaw | aileron | transition | 0.2 | 2 | 0.431596091 | 0.369286155 | 0.2593840755 | 19.10973873 | 6.893039632 | 10.65466351 | 0.2144254294`
- `C2_c1_plus_aileron_adverse_yaw | aileron | transition | 0.4 | 2 | 0.348412533 | 0.6706455151 | 0.4982387686 | 18.1973319 | 12.98156588 | 12.83705858 | 0.1795433982`
- `C2_c1_plus_aileron_adverse_yaw | aileron | transition | 0.6 | 1 | 0.7420479401 | 0.8434374646 | 1.039523569 | 30.18485111 | 27.04173622 | 23.80359462 | `
- `C2_c1_plus_aileron_adverse_yaw | aileron | transition | 0.8 | 2 | 0.06356692994 | 0.3520689221 | 0.02583771774 | 24.93616998 | 3.375352845 | 4.039439889 | 0.2317581732`
- `C2_c1_plus_aileron_adverse_yaw | aileron | post_stall | 0.6 | 1 | 0.03673478563 | 0.7721791252 | 0.09579365866 | 2.789089091 | 7.958670036 | 21.40253222 | `
- `C2_c1_plus_aileron_adverse_yaw | aileron | post_stall | 1 | 2 | 0.3436908224 | 1.512861866 | 0.3606529226 | 18.16090549 | 35.42893129 | 31.274674 | 0.399050484`
- `C2_c1_plus_aileron_adverse_yaw | elevator | transition | 0.2 | 1 | 0.03070998963 | 0.1938428766 | 0.04985253454 | 23.94031424 | 5.900158222 | 12.2812985 | `
- `C2_c1_plus_aileron_adverse_yaw | elevator | transition | 0.4 | 1 | 0.1454193795 | 0.05190277819 | 0.07947425735 | 24.38553704 | 15.67538243 | 2.44273692 | `
- `C2_c1_plus_aileron_adverse_yaw | elevator | transition | 0.6 | 1 | 0.03934116528 | 0.5815338217 | 0.01857658997 | 13.41742582 | 10.55319411 | 4.821640108 | `
- `C2_c1_plus_aileron_adverse_yaw | elevator | transition | 0.8 | 1 | 0.1587742928 | 0.5680059618 | 0.04533238147 | 13.58822376 | 17.57895223 | 2.891770581 | `
- `C2_c1_plus_aileron_adverse_yaw | elevator | transition | 1 | 1 | 0.1013597988 | 0.3510242772 | 0.2650384008 | 0.7193092427 | 3.759516568 | 5.008987048 | `
- `C2_c1_plus_aileron_adverse_yaw | elevator | post_stall | 0.2 | 1 | 0.04773793034 | 0.5720180782 | 0.008904509661 | 13.21487975 | 11.33360967 | 11.16885113 | `
- `C2_c1_plus_aileron_adverse_yaw | elevator | post_stall | 0.4 | 1 | 0.4124999572 | 0.1803021919 | 0.1398588682 | 20.18928767 | 31.18559279 | 5.930292536 | `
- `C2_c1_plus_aileron_adverse_yaw | elevator | post_stall | 0.6 | 1 | 0.1982575792 | 0.3663681206 | 0.07227879954 | 17.3135541 | 26.82477811 | 12.3478575 | `
- `C2_c1_plus_aileron_adverse_yaw | elevator | post_stall | 0.8 | 1 | 0.1960426855 | 0.6496815002 | 0.163377586 | 12.52227733 | 35.13387609 | 8.997510651 | `
- `C2_c1_plus_aileron_adverse_yaw | elevator | post_stall | 1 | 1 | 0.06764185979 | 0.2573780277 | 0.03589197592 | 4.452799237 | 34.87399141 | 10.95048231 | `
- `C2_c1_plus_aileron_adverse_yaw | rudder | transition | 0.2 | 2 | 0.3809946242 | 0.1102183267 | 0.3711026717 | 17.20091547 | 9.146012548 | 6.414536254 | 0.8472201067`
- `C2_c1_plus_aileron_adverse_yaw | rudder | transition | 0.4 | 2 | 0.1366673066 | 0.3763509402 | 0.07934436928 | 20.30980673 | 12.24203892 | 7.047815213 | 0.4839540972`
- `C2_c1_plus_aileron_adverse_yaw | rudder | transition | 0.6 | 2 | 0.1315641988 | 0.4105132777 | 0.02337024279 | 13.17564231 | 2.292803702 | 5.036221087 | 0.1644326789`
- `C2_c1_plus_aileron_adverse_yaw | rudder | transition | 0.8 | 2 | 0.2585862069 | 0.6319623159 | 0.1407733397 | 11.25739586 | 15.00443122 | 16.81721466 | 0.2799716288`
- `C2_c1_plus_aileron_adverse_yaw | rudder | transition | 1 | 2 | 0.4638198473 | 0.7112742487 | 0.1658511882 | 16.30917602 | 4.677019034 | 9.617929843 | 0.02659177764`
- `C3_c1_plus_rudder_roll | aileron | transition | 0.2 | 2 | 0.4499052779 | 0.2385401049 | 0.2368853984 | 20.28372936 | 6.598700005 | 6.685003006 | 0.2948156863`
- `C3_c1_plus_rudder_roll | aileron | transition | 0.4 | 2 | 0.3738235596 | 0.5000816506 | 0.4664340154 | 11.41963098 | 10.74652327 | 5.702543456 | 0.3357760766`
- `C3_c1_plus_rudder_roll | aileron | transition | 0.6 | 1 | 0.8644254084 | 0.04103804474 | 0.935824806 | 28.24866505 | 15.18903256 | 9.92364156 | `
- `C3_c1_plus_rudder_roll | aileron | transition | 0.8 | 2 | 0.106013837 | 0.1095272163 | 0.02243142495 | 24.83714723 | 5.34580526 | 34.10127681 | 0.3311265956`
- `C3_c1_plus_rudder_roll | aileron | post_stall | 0.6 | 1 | 0.02258546157 | 0.4559191382 | 0.117964632 | 17.173813 | 5.515410421 | 51.58472659 | `
- `C3_c1_plus_rudder_roll | aileron | post_stall | 1 | 2 | 0.5398570444 | 0.8202083576 | 0.280073414 | 31.55088453 | 27.59926382 | 59.01953457 | 0.159993069`
- `C3_c1_plus_rudder_roll | elevator | transition | 0.2 | 1 | 0.03070998963 | 0.1938428766 | 0.04985253454 | 23.94031424 | 5.900158222 | 12.2812985 | `
- `C3_c1_plus_rudder_roll | elevator | transition | 0.4 | 1 | 0.1454193795 | 0.05190277819 | 0.07947425735 | 24.38553704 | 15.67538243 | 2.44273692 | `
- `C3_c1_plus_rudder_roll | elevator | transition | 0.6 | 1 | 0.03934116528 | 0.5815338217 | 0.01857658997 | 13.41742582 | 10.55319411 | 4.821640108 | `
- `C3_c1_plus_rudder_roll | elevator | transition | 0.8 | 1 | 0.1587742928 | 0.5680059618 | 0.04533238147 | 13.58822376 | 17.57895223 | 2.891770581 | `
- `C3_c1_plus_rudder_roll | elevator | transition | 1 | 1 | 0.1013597988 | 0.3510242772 | 0.2650384008 | 0.7193092427 | 3.759516568 | 5.008987048 | `
- `C3_c1_plus_rudder_roll | elevator | post_stall | 0.2 | 1 | 0.04773793034 | 0.5720180782 | 0.008904509661 | 13.21487975 | 11.33360967 | 11.16885113 | `
- `C3_c1_plus_rudder_roll | elevator | post_stall | 0.4 | 1 | 0.4124999572 | 0.1803021919 | 0.1398588682 | 20.18928767 | 31.18559279 | 5.930292536 | `
- `C3_c1_plus_rudder_roll | elevator | post_stall | 0.6 | 1 | 0.1982575792 | 0.3663681206 | 0.07227879954 | 17.3135541 | 26.82477811 | 12.3478575 | `
- `C3_c1_plus_rudder_roll | elevator | post_stall | 0.8 | 1 | 0.1960426855 | 0.6496815002 | 0.163377586 | 12.52227733 | 35.13387609 | 8.997510651 | `
- `C3_c1_plus_rudder_roll | elevator | post_stall | 1 | 1 | 0.06764185979 | 0.2573780277 | 0.03589197592 | 4.452799237 | 34.87399141 | 10.95048231 | `
- `C3_c1_plus_rudder_roll | rudder | transition | 0.2 | 2 | 0.3810974222 | 0.109584676 | 0.3707244022 | 17.2292739 | 9.135926404 | 6.425554725 | 0.8476040817`
- `C3_c1_plus_rudder_roll | rudder | transition | 0.4 | 2 | 0.1367176652 | 0.3773231935 | 0.07910522839 | 20.60983697 | 12.30107072 | 7.115453518 | 0.4849555384`
- `C3_c1_plus_rudder_roll | rudder | transition | 0.6 | 2 | 0.1321142482 | 0.4083464675 | 0.02366323501 | 13.41066527 | 2.375779494 | 5.440561928 | 0.1682458962`
- `C3_c1_plus_rudder_roll | rudder | transition | 0.8 | 2 | 0.2591153834 | 0.6286183542 | 0.1389653476 | 11.67459919 | 14.78656406 | 16.6771316 | 0.2704752419`
- `C3_c1_plus_rudder_roll | rudder | transition | 1 | 2 | 0.463489549 | 0.7024146063 | 0.1630722954 | 15.58301474 | 4.980616438 | 9.72773193 | 0.02362262346`
- `C4_c1_plus_surface_side_force | aileron | transition | 0.2 | 2 | 0.4498795001 | 0.2044984381 | 0.2293752306 | 20.71998227 | 6.8275441 | 5.960444297 | 0.288664469`
- `C4_c1_plus_surface_side_force | aileron | transition | 0.4 | 2 | 0.3779795534 | 0.4599998437 | 0.4442656017 | 12.47423234 | 10.35942087 | 9.244408401 | 0.3249808636`
- `C4_c1_plus_surface_side_force | aileron | transition | 0.6 | 1 | 0.8455377114 | 0.3407034814 | 0.8672050008 | 30.78944686 | 12.98772087 | 16.51428863 | `
- `C4_c1_plus_surface_side_force | aileron | transition | 0.8 | 2 | 0.1230077457 | 0.1140370128 | 0.01477109858 | 25.83200664 | 5.538313711 | 37.86257832 | 0.3201840099`
- `C4_c1_plus_surface_side_force | aileron | post_stall | 0.6 | 1 | 0.05292150774 | 0.3017100387 | 0.1449045262 | 17.48929427 | 5.407143413 | 55.11593712 | `
- `C4_c1_plus_surface_side_force | aileron | post_stall | 1 | 2 | 0.5682603513 | 0.5296370026 | 0.2417822588 | 32.73511396 | 25.99152997 | 66.06615494 | 0.1455755684`
- `C4_c1_plus_surface_side_force | elevator | transition | 0.2 | 1 | 0.03070998963 | 0.1938428766 | 0.04985253454 | 23.94031424 | 5.900158222 | 12.2812985 | `
- `C4_c1_plus_surface_side_force | elevator | transition | 0.4 | 1 | 0.1454193795 | 0.05190277819 | 0.07947425735 | 24.38553704 | 15.67538243 | 2.44273692 | `
- `C4_c1_plus_surface_side_force | elevator | transition | 0.6 | 1 | 0.03934116528 | 0.5815338217 | 0.01857658997 | 13.41742582 | 10.55319411 | 4.821640108 | `
- `C4_c1_plus_surface_side_force | elevator | transition | 0.8 | 1 | 0.1587742928 | 0.5680059618 | 0.04533238147 | 13.58822376 | 17.57895223 | 2.891770581 | `
- `C4_c1_plus_surface_side_force | elevator | transition | 1 | 1 | 0.1013597988 | 0.3510242772 | 0.2650384008 | 0.7193092427 | 3.759516568 | 5.008987048 | `
- `C4_c1_plus_surface_side_force | elevator | post_stall | 0.2 | 1 | 0.04773793034 | 0.5720180782 | 0.008904509661 | 13.21487975 | 11.33360967 | 11.16885113 | `
- `C4_c1_plus_surface_side_force | elevator | post_stall | 0.4 | 1 | 0.4124999572 | 0.1803021919 | 0.1398588682 | 20.18928767 | 31.18559279 | 5.930292536 | `
- `C4_c1_plus_surface_side_force | elevator | post_stall | 0.6 | 1 | 0.1982575792 | 0.3663681206 | 0.07227879954 | 17.3135541 | 26.82477811 | 12.3478575 | `
- `C4_c1_plus_surface_side_force | elevator | post_stall | 0.8 | 1 | 0.1960426855 | 0.6496815002 | 0.163377586 | 12.52227733 | 35.13387609 | 8.997510651 | `
- `C4_c1_plus_surface_side_force | elevator | post_stall | 1 | 1 | 0.06764185979 | 0.2573780277 | 0.03589197592 | 4.452799237 | 34.87399141 | 10.95048231 | `
- `C4_c1_plus_surface_side_force | rudder | transition | 0.2 | 2 | 0.3832947951 | 0.08799160161 | 0.367606968 | 17.41988311 | 9.089182395 | 5.774005984 | 0.8539556942`
- `C4_c1_plus_surface_side_force | rudder | transition | 0.4 | 2 | 0.1373653162 | 0.4115454056 | 0.07685926596 | 20.47977338 | 12.15329822 | 7.36716637 | 0.4492736709`
- `C4_c1_plus_surface_side_force | rudder | transition | 0.6 | 2 | 0.1303025136 | 0.462431731 | 0.02588891581 | 13.21742234 | 2.162672997 | 3.573602696 | 0.1189457574`
- `C4_c1_plus_surface_side_force | rudder | transition | 0.8 | 2 | 0.2663429504 | 0.5647196557 | 0.1275272971 | 11.14933543 | 15.36859088 | 15.7878382 | 0.2097239046`
- `C4_c1_plus_surface_side_force | rudder | transition | 1 | 2 | 0.476450524 | 0.5593420436 | 0.1552655149 | 15.42844529 | 5.281566852 | 6.973323849 | 0.004735492089`
- `C5_c2_plus_surface_side_force | aileron | transition | 0.2 | 2 | 0.4374575961 | 0.3355253853 | 0.2513483213 | 19.58002741 | 6.614055795 | 9.905787765 | 0.2082770013`
- `C5_c2_plus_surface_side_force | aileron | transition | 0.4 | 2 | 0.3658281903 | 0.6324981371 | 0.4756980486 | 18.88508232 | 12.48547547 | 9.037362709 | 0.1691835558`
- `C5_c2_plus_surface_side_force | aileron | transition | 0.6 | 1 | 0.8141244442 | 0.5528863456 | 0.9740000645 | 33.4690482 | 25.32062245 | 16.37998084 | `
- `C5_c2_plus_surface_side_force | aileron | transition | 0.8 | 2 | 0.06212448514 | 0.2002504458 | 0.003126695438 | 25.57181012 | 3.617074176 | 8.909299954 | 0.2205014111`
- `C5_c2_plus_surface_side_force | aileron | post_stall | 0.6 | 1 | 0.02616276359 | 0.6115811006 | 0.1316463048 | 2.150588624 | 7.556901382 | 26.6902795 | `
- `C5_c2_plus_surface_side_force | aileron | post_stall | 1 | 2 | 0.4051015417 | 1.220378789 | 0.3329694284 | 19.74026247 | 33.46323667 | 30.40776956 | 0.3927007558`
- `C5_c2_plus_surface_side_force | elevator | transition | 0.2 | 1 | 0.03070998963 | 0.1938428766 | 0.04985253454 | 23.94031424 | 5.900158222 | 12.2812985 | `
- `C5_c2_plus_surface_side_force | elevator | transition | 0.4 | 1 | 0.1454193795 | 0.05190277819 | 0.07947425735 | 24.38553704 | 15.67538243 | 2.44273692 | `
- `C5_c2_plus_surface_side_force | elevator | transition | 0.6 | 1 | 0.03934116528 | 0.5815338217 | 0.01857658997 | 13.41742582 | 10.55319411 | 4.821640108 | `
- `C5_c2_plus_surface_side_force | elevator | transition | 0.8 | 1 | 0.1587742928 | 0.5680059618 | 0.04533238147 | 13.58822376 | 17.57895223 | 2.891770581 | `
- `C5_c2_plus_surface_side_force | elevator | transition | 1 | 1 | 0.1013597988 | 0.3510242772 | 0.2650384008 | 0.7193092427 | 3.759516568 | 5.008987048 | `
- `C5_c2_plus_surface_side_force | elevator | post_stall | 0.2 | 1 | 0.04773793034 | 0.5720180782 | 0.008904509661 | 13.21487975 | 11.33360967 | 11.16885113 | `
- `C5_c2_plus_surface_side_force | elevator | post_stall | 0.4 | 1 | 0.4124999572 | 0.1803021919 | 0.1398588682 | 20.18928767 | 31.18559279 | 5.930292536 | `
- `C5_c2_plus_surface_side_force | elevator | post_stall | 0.6 | 1 | 0.1982575792 | 0.3663681206 | 0.07227879954 | 17.3135541 | 26.82477811 | 12.3478575 | `
- `C5_c2_plus_surface_side_force | elevator | post_stall | 0.8 | 1 | 0.1960426855 | 0.6496815002 | 0.163377586 | 12.52227733 | 35.13387609 | 8.997510651 | `
- `C5_c2_plus_surface_side_force | elevator | post_stall | 1 | 1 | 0.06764185979 | 0.2573780277 | 0.03589197592 | 4.452799237 | 34.87399141 | 10.95048231 | `
- `C5_c2_plus_surface_side_force | rudder | transition | 0.2 | 2 | 0.3832947951 | 0.08799160161 | 0.367606968 | 17.41988311 | 9.089182395 | 5.774005984 | 0.8539556942`
- `C5_c2_plus_surface_side_force | rudder | transition | 0.4 | 2 | 0.1373653162 | 0.4115454056 | 0.07685926596 | 20.47977338 | 12.15329822 | 7.36716637 | 0.4492736709`
- `C5_c2_plus_surface_side_force | rudder | transition | 0.6 | 2 | 0.1303025136 | 0.462431731 | 0.02588891581 | 13.21742234 | 2.162672997 | 3.573602696 | 0.1189457574`
- `C5_c2_plus_surface_side_force | rudder | transition | 0.8 | 2 | 0.2663429504 | 0.5647196557 | 0.1275272971 | 11.14933543 | 15.36859088 | 15.7878382 | 0.2097239046`
- `C5_c2_plus_surface_side_force | rudder | transition | 1 | 2 | 0.476450524 | 0.5593420436 | 0.1552655149 | 15.42844529 | 5.281566852 | 6.973323849 | 0.004735492089`
- `C6_alpha_regime_primary_derivatives | aileron | transition | 0.2 | 2 | 0.4502556758 | 0.2379634727 | 0.2372207248 | 19.89267903 | 6.558701121 | 6.746460734 | 0.2861985112`
- `C6_alpha_regime_primary_derivatives | aileron | transition | 0.4 | 2 | 0.3762125634 | 0.4882846858 | 0.4663290116 | 12.95088686 | 10.31805338 | 5.404312852 | 0.357116124`
- `C6_alpha_regime_primary_derivatives | aileron | transition | 0.6 | 1 | 0.8642426991 | 0.04455337571 | 0.935170924 | 27.64739463 | 15.11296703 | 10.07721827 | `
- `C6_alpha_regime_primary_derivatives | aileron | transition | 0.8 | 2 | 0.09999958247 | 0.1149352257 | 0.02337727329 | 33.26798227 | 3.209636081 | 27.5859156 | 0.7828334578`
- `C6_alpha_regime_primary_derivatives | aileron | post_stall | 0.6 | 1 | 0.01373644751 | 0.4771001363 | 0.1196359146 | 26.57490485 | 7.543437024 | 44.71818339 | `
- `C6_alpha_regime_primary_derivatives | aileron | post_stall | 1 | 2 | 0.5212638927 | 0.8521225422 | 0.2893284035 | 41.2641319 | 28.76991561 | 50.16344838 | 0.1528526349`
- `C6_alpha_regime_primary_derivatives | elevator | transition | 0.2 | 1 | 0.03478639315 | 0.1932747681 | 0.0497926956 | 24.76809284 | 6.142868787 | 12.3198033 | `
- `C6_alpha_regime_primary_derivatives | elevator | transition | 0.4 | 1 | 0.2127776767 | 0.0529769621 | 0.07495645081 | 13.04378791 | 18.30996281 | 0.1466066859 | `
- `C6_alpha_regime_primary_derivatives | elevator | transition | 0.6 | 1 | 0.03413505332 | 0.5832469535 | 0.05340764831 | 13.32307844 | 1.699912643 | 4.295465117 | `
- `C6_alpha_regime_primary_derivatives | elevator | transition | 0.8 | 1 | 0.1282482446 | 0.5708892201 | 0.2752547581 | 13.80023932 | 23.21104265 | 1.729965157 | `
- `C6_alpha_regime_primary_derivatives | elevator | transition | 1 | 1 | 0.1074905821 | 0.2709696817 | 0.3065198329 | 64.62561905 | 42.02229353 | 73.85815283 | `
- `C6_alpha_regime_primary_derivatives | elevator | post_stall | 0.2 | 1 | 0.0153252721 | 0.5796434046 | 0.05576747909 | 13.80041183 | 1.016003445 | 9.894709205 | `
- `C6_alpha_regime_primary_derivatives | elevator | post_stall | 0.4 | 1 | 0.4222573969 | 0.1778912672 | 0.1119192418 | 18.6935574 | 18.16481826 | 2.766856041 | `
- `C6_alpha_regime_primary_derivatives | elevator | post_stall | 0.6 | 1 | 0.1481936047 | 0.3809863728 | 0.1636060119 | 19.44118481 | 3.821803226 | 7.780178104 | `
- `C6_alpha_regime_primary_derivatives | elevator | post_stall | 0.8 | 1 | 0.1321627604 | 0.6712584956 | 0.1921242991 | 12.01535854 | 3.869966089 | 9.904453233 | `
- `C6_alpha_regime_primary_derivatives | elevator | post_stall | 1 | 1 | 0.5313063141 | 0.2735370586 | 0.2247132856 | 0.4032453821 | 10.14285156 | 5.066304314 | `
- `C6_alpha_regime_primary_derivatives | rudder | transition | 0.2 | 2 | 0.3854192832 | 0.08690231303 | 0.3674093163 | 17.21891927 | 8.926235866 | 4.884722053 | 0.8774446713`
- `C6_alpha_regime_primary_derivatives | rudder | transition | 0.4 | 2 | 0.137242525 | 0.4015762926 | 0.07764597855 | 21.7754212 | 12.2794913 | 8.393971732 | 0.4423089708`
- `C6_alpha_regime_primary_derivatives | rudder | transition | 0.6 | 2 | 0.1316073544 | 0.4099332038 | 0.02337717159 | 13.15952017 | 2.285037952 | 5.10684082 | 0.1671076932`
- `C6_alpha_regime_primary_derivatives | rudder | transition | 0.8 | 2 | 0.3141928423 | 0.5464149025 | 0.1150765735 | 14.8258682 | 9.063061768 | 3.698928569 | 0.5918729704`
- `C6_alpha_regime_primary_derivatives | rudder | transition | 1 | 2 | 0.5673228487 | 0.3783267085 | 0.1376945819 | 18.62050786 | 7.539369079 | 16.82286917 | 0.8134279848`
- `C7_c6_plus_alpha_regime_aileron_yaw | aileron | transition | 0.2 | 2 | 0.4479557487 | 0.2758033027 | 0.2448982477 | 18.6294834 | 6.365551024 | 6.761134036 | 0.2502870649`
- `C7_c6_plus_alpha_regime_aileron_yaw | aileron | transition | 0.4 | 2 | 0.3765806117 | 0.4242663483 | 0.481521942 | 19.41921942 | 10.53008228 | 8.394864516 | 0.1903896112`
- `C7_c6_plus_alpha_regime_aileron_yaw | aileron | transition | 0.6 | 1 | 0.873200673 | 0.2608260517 | 0.9726973615 | 28.81427275 | 19.06123795 | 1.278141247 | `
- `C7_c6_plus_alpha_regime_aileron_yaw | aileron | transition | 0.8 | 2 | 0.1120690698 | 0.53817495 | 0.05175475305 | 26.84284691 | 1.806469865 | 15.85761929 | 0.2391260345`
- `C7_c6_plus_alpha_regime_aileron_yaw | aileron | post_stall | 0.6 | 1 | 0.01280430447 | 0.9359750494 | 0.06154110181 | 3.888936326 | 11.70303879 | 6.855344934 | `
- `C7_c6_plus_alpha_regime_aileron_yaw | aileron | post_stall | 1 | 2 | 0.5104491244 | 1.47132691 | 0.2886558009 | 15.68095878 | 32.91726559 | 2.716585989 | 0.1543882408`
- `C7_c6_plus_alpha_regime_aileron_yaw | elevator | transition | 0.2 | 1 | 0.03478639315 | 0.1932747681 | 0.0497926956 | 24.76809284 | 6.142868787 | 12.3198033 | `
- `C7_c6_plus_alpha_regime_aileron_yaw | elevator | transition | 0.4 | 1 | 0.2127776767 | 0.0529769621 | 0.07495645081 | 13.04378791 | 18.30996281 | 0.1466066859 | `
- `C7_c6_plus_alpha_regime_aileron_yaw | elevator | transition | 0.6 | 1 | 0.03413505332 | 0.5832469535 | 0.05340764831 | 13.32307844 | 1.699912643 | 4.295465117 | `
- `C7_c6_plus_alpha_regime_aileron_yaw | elevator | transition | 0.8 | 1 | 0.1282482446 | 0.5708892201 | 0.2752547581 | 13.80023932 | 23.21104265 | 1.729965157 | `
- `C7_c6_plus_alpha_regime_aileron_yaw | elevator | transition | 1 | 1 | 0.1074905821 | 0.2709696817 | 0.3065198329 | 64.62561905 | 42.02229353 | 73.85815283 | `
- `C7_c6_plus_alpha_regime_aileron_yaw | elevator | post_stall | 0.2 | 1 | 0.0153252721 | 0.5796434046 | 0.05576747909 | 13.80041183 | 1.016003445 | 9.894709205 | `
- `C7_c6_plus_alpha_regime_aileron_yaw | elevator | post_stall | 0.4 | 1 | 0.4222573969 | 0.1778912672 | 0.1119192418 | 18.6935574 | 18.16481826 | 2.766856041 | `
- `C7_c6_plus_alpha_regime_aileron_yaw | elevator | post_stall | 0.6 | 1 | 0.1481936047 | 0.3809863728 | 0.1636060119 | 19.44118481 | 3.821803226 | 7.780178104 | `
- `C7_c6_plus_alpha_regime_aileron_yaw | elevator | post_stall | 0.8 | 1 | 0.1321627604 | 0.6712584956 | 0.1921242991 | 12.01535854 | 3.869966089 | 9.904453233 | `
- `C7_c6_plus_alpha_regime_aileron_yaw | elevator | post_stall | 1 | 1 | 0.5313063141 | 0.2735370586 | 0.2247132856 | 0.4032453821 | 10.14285156 | 5.066304314 | `
- `C7_c6_plus_alpha_regime_aileron_yaw | rudder | transition | 0.2 | 2 | 0.3854192832 | 0.08690231303 | 0.3674093163 | 17.21891927 | 8.926235866 | 4.884722053 | 0.8774446713`
- `C7_c6_plus_alpha_regime_aileron_yaw | rudder | transition | 0.4 | 2 | 0.137242525 | 0.4015762926 | 0.07764597855 | 21.7754212 | 12.2794913 | 8.393971732 | 0.4423089708`
- `C7_c6_plus_alpha_regime_aileron_yaw | rudder | transition | 0.6 | 2 | 0.1316073544 | 0.4099332038 | 0.02337717159 | 13.15952017 | 2.285037952 | 5.10684082 | 0.1671076932`
- `C7_c6_plus_alpha_regime_aileron_yaw | rudder | transition | 0.8 | 2 | 0.3141928423 | 0.5464149025 | 0.1150765735 | 14.8258682 | 9.063061768 | 3.698928569 | 0.5918729704`
- `C7_c6_plus_alpha_regime_aileron_yaw | rudder | transition | 1 | 2 | 0.5673228487 | 0.3783267085 | 0.1376945819 | 18.62050786 | 7.539369079 | 16.82286917 | 0.8134279848`
- `C8_c7_plus_alpha_regime_aileron_side_force | aileron | transition | 0.2 | 2 | 0.439175196 | 0.368571441 | 0.2624594899 | 17.83434139 | 6.183186632 | 9.036051052 | 0.2598258178`
- `C8_c7_plus_alpha_regime_aileron_side_force | aileron | transition | 0.4 | 2 | 0.3532878779 | 0.7849032767 | 0.5026444447 | 16.56605298 | 11.05752438 | 6.900845019 | 0.2296166847`
- `C8_c7_plus_alpha_regime_aileron_side_force | aileron | transition | 0.6 | 1 | 0.8057395803 | 0.8168252442 | 1.071434083 | 23.41312964 | 21.91893116 | 14.37350525 | `
- `C8_c7_plus_alpha_regime_aileron_side_force | aileron | transition | 0.8 | 2 | 0.07332288198 | 0.1722035052 | 0.06847311473 | 27.39691404 | 1.882770707 | 4.329228069 | 0.2036716899`
- `C8_c7_plus_alpha_regime_aileron_side_force | aileron | post_stall | 0.6 | 1 | 0.008916985087 | 0.3263544532 | 0.2186946139 | 4.191550578 | 10.219897 | 23.0680352 | `
- `C8_c7_plus_alpha_regime_aileron_side_force | aileron | post_stall | 1 | 2 | 0.3823752805 | 1.127031366 | 0.5507969794 | 12.27764433 | 33.85331099 | 18.28869259 | 0.1975814569`
- `C8_c7_plus_alpha_regime_aileron_side_force | elevator | transition | 0.2 | 1 | 0.03478639315 | 0.1932747681 | 0.0497926956 | 24.76809284 | 6.142868787 | 12.3198033 | `
- `C8_c7_plus_alpha_regime_aileron_side_force | elevator | transition | 0.4 | 1 | 0.2127776767 | 0.0529769621 | 0.07495645081 | 13.04378791 | 18.30996281 | 0.1466066859 | `
- `C8_c7_plus_alpha_regime_aileron_side_force | elevator | transition | 0.6 | 1 | 0.03413505332 | 0.5832469535 | 0.05340764831 | 13.32307844 | 1.699912643 | 4.295465117 | `
- `C8_c7_plus_alpha_regime_aileron_side_force | elevator | transition | 0.8 | 1 | 0.1282482446 | 0.5708892201 | 0.2752547581 | 13.80023932 | 23.21104265 | 1.729965157 | `
- `C8_c7_plus_alpha_regime_aileron_side_force | elevator | transition | 1 | 1 | 0.1074905821 | 0.2709696817 | 0.3065198329 | 64.62561905 | 42.02229353 | 73.85815283 | `
- `C8_c7_plus_alpha_regime_aileron_side_force | elevator | post_stall | 0.2 | 1 | 0.0153252721 | 0.5796434046 | 0.05576747909 | 13.80041183 | 1.016003445 | 9.894709205 | `
- `C8_c7_plus_alpha_regime_aileron_side_force | elevator | post_stall | 0.4 | 1 | 0.4222573969 | 0.1778912672 | 0.1119192418 | 18.6935574 | 18.16481826 | 2.766856041 | `
- `C8_c7_plus_alpha_regime_aileron_side_force | elevator | post_stall | 0.6 | 1 | 0.1481936047 | 0.3809863728 | 0.1636060119 | 19.44118481 | 3.821803226 | 7.780178104 | `
- `C8_c7_plus_alpha_regime_aileron_side_force | elevator | post_stall | 0.8 | 1 | 0.1321627604 | 0.6712584956 | 0.1921242991 | 12.01535854 | 3.869966089 | 9.904453233 | `
- `C8_c7_plus_alpha_regime_aileron_side_force | elevator | post_stall | 1 | 1 | 0.5313063141 | 0.2735370586 | 0.2247132856 | 0.4032453821 | 10.14285156 | 5.066304314 | `
- `C8_c7_plus_alpha_regime_aileron_side_force | rudder | transition | 0.2 | 2 | 0.3854192832 | 0.08690231303 | 0.3674093163 | 17.21891927 | 8.926235866 | 4.884722053 | 0.8774446713`
- `C8_c7_plus_alpha_regime_aileron_side_force | rudder | transition | 0.4 | 2 | 0.137242525 | 0.4015762926 | 0.07764597855 | 21.7754212 | 12.2794913 | 8.393971732 | 0.4423089708`
- `C8_c7_plus_alpha_regime_aileron_side_force | rudder | transition | 0.6 | 2 | 0.1316073544 | 0.4099332038 | 0.02337717159 | 13.15952017 | 2.285037952 | 5.10684082 | 0.1671076932`
- `C8_c7_plus_alpha_regime_aileron_side_force | rudder | transition | 0.8 | 2 | 0.3141928423 | 0.5464149025 | 0.1150765735 | 14.8258682 | 9.063061768 | 3.698928569 | 0.5918729704`
- `C8_c7_plus_alpha_regime_aileron_side_force | rudder | transition | 1 | 2 | 0.5673228487 | 0.3783267085 | 0.1376945819 | 18.62050786 | 7.539369079 | 16.82286917 | 0.8134279848`

## 7. Launch-Confidence Diagnostic

Launch confidence is a diagnostic weight and grouping variable, not a new acceptance gate. It reuses the neutral SysID lateral-contamination strategy with reference `phi0=psi0=v0=p0=r0=0`, so the study can test whether real-vs-replay mismatch is launch-condition driven.

- all successful replays: `174` total, `139` high-confidence, `35` medium-confidence, `0` low-confidence; mean confidence weight `0.8246298746`, mean lateral-contamination score `0.3490437321`
- primary antisymmetric residual check; lower is better, negative delta means the confidence subset reduced mismatch:
- aileron: all `0.170756809`, high-confidence `0.2536082227` (delta `0.08285141368`), weighted `0.1756967213` (delta `0.004939912343`)
- elevator: all `1.030732967`, high-confidence `1.025348524` (delta `-0.005384443176`), weighted `1.017842185` (delta `-0.01289078237`)
- rudder: all `0.4431842749`, high-confidence `0.4293002134` (delta `-0.01388406144`), weighted `0.4420018205` (delta `-0.001182454419`)

## 8. Aileron Effectiveness

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

## 9. Elevator Effectiveness

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

## 10. Rudder Effectiveness

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

## 11. Cross-Coupling Observations

Aileron yaw response and rudder roll response are reported as diagnostic coupling evidence. They are not promoted as lateral transition aerodynamic derivatives by this study.

## 12. Symmetric Launch/Trim Contamination

Symmetric response is separated from antisymmetric response. Large symmetric terms are interpreted as launch, trim, hardware, or model-mismatch contamination rather than hidden inside a surface effectiveness scale.

- aileron: mean absolute primary symmetric response `0.3904063872`
- elevator: mean absolute primary symmetric response `1.981335105`
- rudder: mean absolute primary symmetric response `0.3077712386`

## 13. Optional Surface/Aero Fit Result

- `S0_frozen_neutral`: `evaluated_frozen_active_neutral_model`, promoted `False`
- `S1_surface_effectiveness_scales`: `not_run_disabled_by_cli`, promoted `False`
- `S2_scales_plus_neutral_biases`: `not_run_disabled_by_cli`, promoted `False`
- `D0_launch_confidence_weighted_derivative_fit_basis`: `diagnostic_derivative_level_fit_not_promoted`, promoted `False`
- S1/S2 surface-scale diagnostics are disabled by default because measured surface magnitudes are used.

- The derivative diagnostic fits residual control force/moment coefficients from measured acceleration with launch-confidence weighting; it is not replay-promoted.
- `@normal`, `@transition`, and `@post_stall` rows are alpha-regime diagnostics. Rudder post-stall is intentionally not fitted independently and falls back to transition in scheduled replay candidates.
- `CY_delta_a_residual`: coeff `0.4798183808`, held-out baseline `0.2512200618`, candidate `0.226674254`, improved `True`
- `CY_delta_a_residual@normal`: coeff `-0.8665377195`, held-out baseline `0.1378775225`, candidate `0.1714494504`, improved `False`
- `CY_delta_a_residual@transition`: coeff `0.3179744124`, held-out baseline `0.2314142842`, candidate `0.2148967921`, improved `True`
- `CY_delta_a_residual@post_stall`: coeff `3.037836153`, held-out baseline `0.8345970198`, candidate `0.2587670892`, improved `True`
- `Cl_delta_a_residual`: coeff `0.05628318768`, held-out baseline `0.01936537792`, candidate `0.01595728473`, improved `True`
- `Cl_delta_a_residual@normal`: coeff `0.05942017802`, held-out baseline `0.0132122793`, candidate `0.01358156956`, improved `False`
- `Cl_delta_a_residual@transition`: coeff `0.04509569964`, held-out baseline `0.018790668`, candidate `0.01586562353`, improved `True`
- `Cl_delta_a_residual@post_stall`: coeff `0.1367980844`, held-out baseline `0.04585556318`, candidate `0.009175935016`, improved `True`
- `Cn_delta_a_residual`: coeff `-0.08210197202`, held-out baseline `0.01912805222`, candidate `0.01377962783`, improved `True`
- `Cn_delta_a_residual@normal`: coeff `-0.02771465871`, held-out baseline `0.009773210532`, candidate `0.009051719248`, improved `True`
- `Cn_delta_a_residual@transition`: coeff `-0.07922865127`, held-out baseline `0.01899243757`, candidate `0.01367764121`, improved `True`
- `Cn_delta_a_residual@post_stall`: coeff `-0.1580337823`, held-out baseline `0.05176281053`, candidate `0.01191434733`, improved `True`
- `Cm_delta_e_residual`: coeff `-0.115511593`, held-out baseline `0.09140446186`, candidate `0.0845325132`, improved `True`
- `Cm_delta_e_residual@normal`: coeff `0.02261583888`, held-out baseline `0.0994506879`, candidate `0.09821632403`, improved `True`
- `Cm_delta_e_residual@transition`: coeff `-0.09352878291`, held-out baseline `0.06978945378`, candidate `0.06561027273`, improved `True`
- `Cm_delta_e_residual@post_stall`: coeff `-0.4175659564`, held-out baseline `0.1251490743`, candidate `0.05613541606`, improved `True`
- `CY_delta_r_residual`: coeff `-0.1708085654`, held-out baseline `0.2371454031`, candidate `0.2394389883`, improved `False`
- `Cn_delta_r_residual`: coeff `-0.01043556406`, held-out baseline `0.01338416377`, candidate `0.01290536735`, improved `True`
- `Cn_delta_r_residual@normal`: coeff `-0.0264172626`, held-out baseline `0.01219381698`, candidate `0.01428393127`, improved `False`
- `Cn_delta_r_residual@transition`: coeff `-0.01021342626`, held-out baseline `0.01346449988`, candidate `0.01292318496`, improved `True`
- `Cl_delta_r_residual`: coeff `-0.004294737192`, held-out baseline `0.01286848089`, candidate `0.01276270234`, improved `True`

## 14. Promotion Decision

No model parameter is promoted by this analysis. A surface-only update would require held-out deflection improvement, neutral replay preservation, interpretable signs/magnitudes, and closed-loop smoke evidence.

## 15. Limitations

- Launch-condition contamination remains visible in the symmetric response.
- Deflection data are sustained pulse-ladder throws, not a broad aero excitation design.
- Candidate derivative rows are diagnostic summaries, not checked-in plant changes.
- Alpha-regime candidate rows are diagnostic; they do not establish a validated surface-effectiveness schedule.
- Regime-ladder rows are evidence reporting cells, not independent pass gates.
- S1/S2 surface-scale rows are disabled by default because measured surface magnitudes are already used.
- R5/R7/R8/R10/R11 semantics are unchanged.

## 16. Reproducibility Commands

```powershell
python 03_Control/02_Inner_Loop/run_control_surface_effectiveness_study.py
pytest 03_Control/tests/test_control_surface_effectiveness_study.py
pytest 03_Control/tests/test_neutral_aero_residual_sysid_contract.py
```
