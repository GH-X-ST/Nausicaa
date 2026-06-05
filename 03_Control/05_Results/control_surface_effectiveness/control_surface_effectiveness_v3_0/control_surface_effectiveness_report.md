# Real-Flight Control Surface Effectiveness Study v3.0

## 1. Purpose and Claim Boundary

The neutral aero fit remains frozen. Deflection ladder throws support one conservative elevator aerodynamic-effectiveness correction in the active plant model; aileron, rudder, and cross-axis terms remain diagnostic. This is not broad aerodynamic SysID and does not claim accurate full 6-DoF lateral derivative identification.

- active calibrated model: `neutral_dry_air_residual_calibrated_replay_n30_compact_coupled_elevator_effectiveness_v1`
- claim boundary: `neutral aero replay alignment plus conservative elevator-only surface effectiveness; aileron/rudder and lateral/coupling terms diagnostic only; no broad aero SysID`
- promotion decision: `promoted_conservative_elevator_effectiveness_only`
- active surface-effectiveness scales: `{'delta_a_aero_effectiveness_scale': 1.0, 'delta_e_aero_effectiveness_scale': 0.6, 'delta_r_aero_effectiveness_scale': 1.0}`

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

Each usable throw is replayed from its measured launch state using the active calibrated model, logged command schedule, nominal command-onset delay, and actuator lag from the throw manifest. The command conversion and measured surface-angle convention are unchanged; the promoted elevator update scales only the aerodynamic effectiveness in the strip model.

- successful replays: `174` / `174`
- replay workers: `8` / max `8`
- replay dx MAE: `0.2427547947` m
- replay dy MAE: `0.4871987658` m
- replay altitude-loss MAE: `0.2185921058` m

## 5. Candidate Replay Error Summary

The candidate comparison now puts simple pairwise +/- response-gain replay first, then keeps the launch-confidence-weighted residual surface aero/coupling derivatives as diagnostic comparisons. C0 is the frozen active calibrated model, including the promoted conservative elevator effectiveness scale. P1-P4 are extra pairwise response-gain replays. C1-C8 are derivative/coupling diagnostics. No aileron, rudder, lateral/coupling, or regime-scheduled candidate is promoted by this report.

- successful candidate-family replays: `2088` / `2088`

- replay MAE comparison; lower is better:
`candidate | split | surface | dx | dy | altitude | phi | theta | psi | primary antisym`
- `C0_frozen_neutral | all | all | 0.2427547947 | 0.4871987658 | 0.2185921058 | 17.98068933 | 12.12375938 | 14.99646591 | 0.2848939036`
- `C0_frozen_neutral | all | aileron | 0.3777756604 | 0.5492673767 | 0.3277644207 | 25.45684136 | 10.31288765 | 22.49422645 | 0.170756809`
- `C0_frozen_neutral | all | elevator | 0.1493012726 | 0.4128 | 0.1740175063 | 13.32789667 | 19.23640566 | 8.745228448 | 0.240740627`
- `C0_frozen_neutral | all | rudder | 0.1972482375 | 0.4971760348 | 0.1513435815 | 14.94821023 | 6.975838602 | 13.51289116 | 0.4431842749`
- `C0_frozen_neutral | heldout | all | 0.2483926588 | 0.4203875944 | 0.1929894716 | 17.05109041 | 13.50012906 | 17.16307894 | 0.4326478474`
- `C0_frozen_neutral | heldout | aileron | 0.3905845536 | 0.3820599909 | 0.3055433691 | 24.09346252 | 11.46736938 | 30.09453911 | 0.1848068848`
- `C0_frozen_neutral | heldout | elevator | 0.1150740906 | 0.3808274752 | 0.1075610908 | 13.08956919 | 19.93385708 | 6.789256229 | 0.5415352502`
- `C0_frozen_neutral | heldout | rudder | 0.2395193321 | 0.498275317 | 0.1658639549 | 13.97023953 | 9.099160718 | 14.6054415 | 0.5716014072`
- `P1_pairwise_aileron_gain | all | all | 0.2397054836 | 0.4878384396 | 0.2101015568 | 16.46079037 | 11.61235242 | 14.08426888 | 0.2924921316`
- `P1_pairwise_aileron_gain | all | aileron | 0.3687827769 | 0.5511538723 | 0.3027244967 | 20.97442746 | 8.804670504 | 19.80401825 | 0.193551493`
- `P1_pairwise_aileron_gain | all | elevator | 0.1493012726 | 0.4128 | 0.1740175063 | 13.32789667 | 19.23640566 | 8.745228448 | 0.240740627`
- `P1_pairwise_aileron_gain | all | rudder | 0.1972482375 | 0.4971760348 | 0.1513435815 | 14.94821023 | 6.975838602 | 13.51289116 | 0.4431842749`
- `P1_pairwise_aileron_gain | heldout | all | 0.241896758 | 0.420746498 | 0.1857038735 | 15.56863452 | 13.30117479 | 16.24582413 | 0.4472389966`
- `P1_pairwise_aileron_gain | heldout | aileron | 0.3710968512 | 0.3831367016 | 0.2836865748 | 19.64609485 | 10.87050657 | 27.34277468 | 0.2285803325`
- `P1_pairwise_aileron_gain | heldout | elevator | 0.1150740906 | 0.3808274752 | 0.1075610908 | 13.08956919 | 19.93385708 | 6.789256229 | 0.5415352502`
- `P1_pairwise_aileron_gain | heldout | rudder | 0.2395193321 | 0.498275317 | 0.1658639549 | 13.97023953 | 9.099160718 | 14.6054415 | 0.5716014072`
- `P2_pairwise_elevator_gain | all | all | 0.2456012605 | 0.4877379254 | 0.2202972755 | 18.11913185 | 11.19916256 | 14.91663953 | 0.2828488065`
- `P2_pairwise_elevator_gain | all | aileron | 0.3777756604 | 0.5492673767 | 0.3277644207 | 25.45684136 | 10.31288765 | 22.49422645 | 0.170756809`
- `P2_pairwise_elevator_gain | all | elevator | 0.1579904839 | 0.4144458556 | 0.1792227612 | 13.75051067 | 16.41395219 | 8.501547898 | 0.2346053357`
- `P2_pairwise_elevator_gain | all | rudder | 0.1972482375 | 0.4971760348 | 0.1513435815 | 14.94821023 | 6.975838602 | 13.51289116 | 0.4431842749`
- `P2_pairwise_elevator_gain | heldout | all | 0.2565929506 | 0.4198693159 | 0.2121000379 | 17.34186766 | 12.64936803 | 17.06660557 | 0.5098985596`
- `P2_pairwise_elevator_gain | heldout | aileron | 0.3905845536 | 0.3820599909 | 0.3055433691 | 24.09346252 | 11.46736938 | 30.09453911 | 0.1848068848`
- `P2_pairwise_elevator_gain | heldout | elevator | 0.1396749661 | 0.3792726397 | 0.1648927896 | 13.96190093 | 17.381574 | 6.499836103 | 0.7732873868`
- `P2_pairwise_elevator_gain | heldout | rudder | 0.2395193321 | 0.498275317 | 0.1658639549 | 13.97023953 | 9.099160718 | 14.6054415 | 0.5716014072`
- `P3_pairwise_rudder_gain | all | all | 0.2562881992 | 0.4795865672 | 0.2210275506 | 18.50220683 | 12.117463 | 12.82210454 | 0.1801114837`
- `P3_pairwise_rudder_gain | all | aileron | 0.3777756604 | 0.5492673767 | 0.3277644207 | 25.45684136 | 10.31288765 | 22.49422645 | 0.170756809`
- `P3_pairwise_rudder_gain | all | elevator | 0.1493012726 | 0.4128 | 0.1740175063 | 13.32789667 | 19.23640566 | 8.745228448 | 0.240740627`
- `P3_pairwise_rudder_gain | all | rudder | 0.2378484512 | 0.474339439 | 0.158649916 | 16.51276273 | 6.956949442 | 6.989807044 | 0.1288370151`
- `P3_pairwise_rudder_gain | heldout | all | 0.2598514582 | 0.417015554 | 0.1915179565 | 17.87143669 | 13.19056703 | 15.08661739 | 0.3291356895`
- `P3_pairwise_rudder_gain | heldout | aileron | 0.3905845536 | 0.3820599909 | 0.3055433691 | 24.09346252 | 11.46736938 | 30.09453911 | 0.1848068848`
- `P3_pairwise_rudder_gain | heldout | elevator | 0.1150740906 | 0.3808274752 | 0.1075610908 | 13.08956919 | 19.93385708 | 6.789256229 | 0.5415352502`
- `P3_pairwise_rudder_gain | heldout | rudder | 0.2738957304 | 0.488159196 | 0.1614494096 | 16.43127835 | 8.170474624 | 8.376056827 | 0.2610649335`
- `P4_pairwise_all_surface_gains | all | all | 0.256085354 | 0.4807654006 | 0.2142421714 | 17.12075039 | 10.68145921 | 11.83008112 | 0.1856646146`
- `P4_pairwise_all_surface_gains | all | aileron | 0.3687827769 | 0.5511538723 | 0.3027244967 | 20.97442746 | 8.804670504 | 19.80401825 | 0.193551493`
- `P4_pairwise_all_surface_gains | all | elevator | 0.1579904839 | 0.4144458556 | 0.1792227612 | 13.75051067 | 16.41395219 | 8.501547898 | 0.2346053357`
- `P4_pairwise_all_surface_gains | all | rudder | 0.2378484512 | 0.474339439 | 0.158649916 | 16.51276273 | 6.956949442 | 6.989807044 | 0.1288370151`
- `P4_pairwise_all_surface_gains | heldout | all | 0.2615558492 | 0.4168561791 | 0.2033429247 | 16.67975805 | 12.14085173 | 14.0728892 | 0.4209775509`
- `P4_pairwise_all_surface_gains | heldout | aileron | 0.3710968512 | 0.3831367016 | 0.2836865748 | 19.64609485 | 10.87050657 | 27.34277468 | 0.2285803325`
- `P4_pairwise_all_surface_gains | heldout | elevator | 0.1396749661 | 0.3792726397 | 0.1648927896 | 13.96190093 | 17.381574 | 6.499836103 | 0.7732873868`
- `P4_pairwise_all_surface_gains | heldout | rudder | 0.2738957304 | 0.488159196 | 0.1614494096 | 16.43127835 | 8.170474624 | 8.376056827 | 0.2610649335`
- `C1_primary_moment_derivatives | all | all | 0.2520120116 | 0.4734177015 | 0.220563676 | 17.59550395 | 11.91328747 | 12.56044672 | 0.2456497059`
- `C1_primary_moment_derivatives | all | aileron | 0.3764298926 | 0.5458803669 | 0.3319113355 | 23.07618194 | 9.629541515 | 19.83635294 | 0.2859409949`
- `C1_primary_moment_derivatives | all | elevator | 0.1490491673 | 0.4129675685 | 0.1745251453 | 13.35258934 | 19.15161217 | 8.736122587 | 0.2356326651`
- `C1_primary_moment_derivatives | all | rudder | 0.2266366176 | 0.4591135691 | 0.1525409922 | 16.19009241 | 7.122882371 | 8.917481347 | 0.2153754577`
- `C1_primary_moment_derivatives | heldout | all | 0.2565877299 | 0.4041226459 | 0.1907902798 | 16.97622054 | 13.56808988 | 14.33556007 | 0.3974451742`
- `C1_primary_moment_derivatives | heldout | aileron | 0.3826210308 | 0.3833671842 | 0.3065437944 | 22.16052622 | 12.12850277 | 27.25250838 | 0.2787030309`
- `C1_primary_moment_derivatives | heldout | elevator | 0.1128157222 | 0.3809369317 | 0.1097386828 | 13.11754812 | 19.90330578 | 6.767428415 | 0.5531984337`
- `C1_primary_moment_derivatives | heldout | rudder | 0.2743264368 | 0.4480638218 | 0.1560883623 | 15.65058728 | 8.672461085 | 8.986743411 | 0.3604340579`
- `C2_c1_plus_aileron_adverse_yaw | all | all | 0.2286735435 | 0.5257844525 | 0.2390034956 | 16.74905508 | 13.60368764 | 10.16687079 | 0.2056650112`
- `C2_c1_plus_aileron_adverse_yaw | all | aileron | 0.3076011901 | 0.7003179038 | 0.3862931765 | 20.5798751 | 14.61478946 | 12.7773324 | 0.1659869107`
- `C2_c1_plus_aileron_adverse_yaw | all | elevator | 0.1490491673 | 0.4129675685 | 0.1745251453 | 13.35258934 | 19.15161217 | 8.736122587 | 0.2356326651`
- `C2_c1_plus_aileron_adverse_yaw | all | rudder | 0.2266366176 | 0.4591135691 | 0.1525409922 | 16.19009241 | 7.122882371 | 8.917481347 | 0.2153754577`
- `C2_c1_plus_aileron_adverse_yaw | heldout | all | 0.2341579023 | 0.523844968 | 0.2027271549 | 16.04878621 | 14.60386181 | 10.67865057 | 0.374652676`
- `C2_c1_plus_aileron_adverse_yaw | heldout | aileron | 0.3153315478 | 0.7425341505 | 0.3423544197 | 19.37822324 | 15.23581855 | 16.28177988 | 0.2103255366`
- `C2_c1_plus_aileron_adverse_yaw | heldout | elevator | 0.1128157222 | 0.3809369317 | 0.1097386828 | 13.11754812 | 19.90330578 | 6.767428415 | 0.5531984337`
- `C2_c1_plus_aileron_adverse_yaw | heldout | rudder | 0.2743264368 | 0.4480638218 | 0.1560883623 | 15.65058728 | 8.672461085 | 8.986743411 | 0.3604340579`
- `C3_c1_plus_rudder_roll | all | all | 0.2520853605 | 0.4730972398 | 0.2203788905 | 17.67443276 | 11.93506543 | 12.54920896 | 0.2468890269`
- `C3_c1_plus_rudder_roll | all | aileron | 0.3764298926 | 0.5458803669 | 0.3319113355 | 23.07618194 | 9.629541515 | 19.83635294 | 0.2859409949`
- `C3_c1_plus_rudder_roll | all | elevator | 0.1490491673 | 0.4129675685 | 0.1745251453 | 13.35258934 | 19.15161217 | 8.736122587 | 0.2356326651`
- `C3_c1_plus_rudder_roll | all | rudder | 0.2268566643 | 0.4581521839 | 0.1519866357 | 16.42687885 | 7.188216248 | 8.883768085 | 0.2190934208`
- `C3_c1_plus_rudder_roll | heldout | all | 0.2566478688 | 0.4031871918 | 0.1904628596 | 16.99318412 | 13.58259999 | 14.36574118 | 0.3969607136`
- `C3_c1_plus_rudder_roll | heldout | aileron | 0.3826210308 | 0.3833671842 | 0.3065437944 | 22.16052622 | 12.12850277 | 27.25250838 | 0.2787030309`
- `C3_c1_plus_rudder_roll | heldout | elevator | 0.1128157222 | 0.3809369317 | 0.1097386828 | 13.11754812 | 19.90330578 | 6.767428415 | 0.5531984337`
- `C3_c1_plus_rudder_roll | heldout | rudder | 0.2745068536 | 0.4452574595 | 0.1551061017 | 15.70147801 | 8.715991423 | 9.077286741 | 0.3589806763`
- `C4_c1_plus_surface_side_force | all | all | 0.2551550135 | 0.4664941324 | 0.2101646166 | 17.7678362 | 11.64985546 | 13.28464642 | 0.2303206256`
- `C4_c1_plus_surface_side_force | all | aileron | 0.3816793375 | 0.5531536448 | 0.3020070436 | 23.97290078 | 8.751928922 | 22.95199868 | 0.2784646893`
- `C4_c1_plus_surface_side_force | all | elevator | 0.1490491673 | 0.4129675685 | 0.1745251453 | 13.35258934 | 19.15161217 | 8.736122587 | 0.2356326651`
- `C4_c1_plus_surface_side_force | all | rudder | 0.2307256707 | 0.4309441824 | 0.1517636971 | 15.79490965 | 7.225330182 | 7.920716672 | 0.1768645225`
- `C4_c1_plus_surface_side_force | heldout | all | 0.261746098 | 0.3746729435 | 0.1825393553 | 17.27888706 | 13.4324054 | 15.21745187 | 0.381918873`
- `C4_c1_plus_surface_side_force | heldout | aileron | 0.393671352 | 0.3258758115 | 0.2872497906 | 23.18014115 | 11.58284816 | 30.98973977 | 0.2652312815`
- `C4_c1_plus_surface_side_force | heldout | elevator | 0.1128157222 | 0.3809369317 | 0.1097386828 | 13.11754812 | 19.90330578 | 6.767428415 | 0.5531984337`
- `C4_c1_plus_surface_side_force | heldout | rudder | 0.2787512198 | 0.4172060875 | 0.1506295924 | 15.53897191 | 8.81106227 | 7.89518742 | 0.3273269038`
- `C5_c2_plus_surface_side_force | all | all | 0.2393314701 | 0.4911621754 | 0.228043305 | 17.05353131 | 13.31522438 | 9.430869288 | 0.1941829223`
- `C5_c2_plus_surface_side_force | all | aileron | 0.3350132941 | 0.6259034666 | 0.354734023 | 21.8663067 | 13.6633559 | 11.58662206 | 0.1700515793`
- `C5_c2_plus_surface_side_force | all | elevator | 0.1490491673 | 0.4129675685 | 0.1745251453 | 13.35258934 | 19.15161217 | 8.736122587 | 0.2356326651`
- `C5_c2_plus_surface_side_force | all | rudder | 0.2307256707 | 0.4309441824 | 0.1517636971 | 15.79490965 | 7.225330182 | 7.920716672 | 0.1768645225`
- `C5_c2_plus_surface_side_force | heldout | all | 0.2432326751 | 0.4641067718 | 0.1945204703 | 16.32464006 | 14.41269628 | 10.20722862 | 0.3619107871`
- `C5_c2_plus_surface_side_force | heldout | aileron | 0.3381310834 | 0.5941772961 | 0.3231931357 | 20.31740015 | 14.52372081 | 15.95907003 | 0.2052070239`
- `C5_c2_plus_surface_side_force | heldout | elevator | 0.1128157222 | 0.3809369317 | 0.1097386828 | 13.11754812 | 19.90330578 | 6.767428415 | 0.5531984337`
- `C5_c2_plus_surface_side_force | heldout | rudder | 0.2787512198 | 0.4172060875 | 0.1506295924 | 15.53897191 | 8.81106227 | 7.89518742 | 0.3273269038`
- `C6_alpha_regime_primary_derivatives | all | all | 0.2721309376 | 0.4612153139 | 0.2285856273 | 18.93798302 | 8.7331715 | 11.28125039 | 0.264024976`
- `C6_alpha_regime_primary_derivatives | all | aileron | 0.3760232314 | 0.5442848505 | 0.334622431 | 26.03556166 | 9.558351287 | 17.97446225 | 0.3831466476`
- `C6_alpha_regime_primary_derivatives | all | elevator | 0.1818802774 | 0.4149092356 | 0.1925118908 | 13.05164984 | 9.576150987 | 8.09795549 | 0.2098580928`
- `C6_alpha_regime_primary_derivatives | all | rudder | 0.2551420117 | 0.4222212417 | 0.1561723783 | 17.502877 | 7.065319119 | 7.60104883 | 0.1990701877`
- `C6_alpha_regime_primary_derivatives | heldout | all | 0.2923559581 | 0.3780426178 | 0.2063865118 | 18.81431202 | 9.951057276 | 12.59049126 | 0.4982993177`
- `C6_alpha_regime_primary_derivatives | heldout | aileron | 0.3773442575 | 0.3908265365 | 0.3087317665 | 26.89736596 | 12.03690164 | 23.45956768 | 0.3924613899`
- `C6_alpha_regime_primary_derivatives | heldout | elevator | 0.1925666461 | 0.3786706329 | 0.1661870446 | 12.42552277 | 9.797630992 | 6.530439619 | 0.524004105`
- `C6_alpha_regime_primary_derivatives | heldout | rudder | 0.3071569707 | 0.3646306841 | 0.1442407244 | 17.12004734 | 8.018639192 | 7.781466469 | 0.5784324581`
- `C7_c6_plus_alpha_regime_aileron_yaw | all | all | 0.2713611614 | 0.48381984 | 0.2369211643 | 16.82164457 | 9.483127335 | 8.413345592 | 0.2097024231`
- `C7_c6_plus_alpha_regime_aileron_yaw | all | aileron | 0.3737530438 | 0.610949046 | 0.3592052012 | 19.79415675 | 11.77008544 | 9.516573524 | 0.2201789888`
- `C7_c6_plus_alpha_regime_aileron_yaw | all | elevator | 0.1818802774 | 0.4149092356 | 0.1925118908 | 13.05164984 | 9.576150987 | 8.09795549 | 0.2098580928`
- `C7_c6_plus_alpha_regime_aileron_yaw | all | rudder | 0.2551420117 | 0.4222212417 | 0.1561723783 | 17.502877 | 7.065319119 | 7.60104883 | 0.1990701877`
- `C7_c6_plus_alpha_regime_aileron_yaw | heldout | all | 0.2925783418 | 0.4682985765 | 0.209072588 | 16.31013091 | 10.40552387 | 7.290431824 | 0.4339335626`
- `C7_c6_plus_alpha_regime_aileron_yaw | heldout | aileron | 0.3780114087 | 0.6615944124 | 0.3167899951 | 19.38482261 | 13.40030143 | 7.559389384 | 0.1993641247`
- `C7_c6_plus_alpha_regime_aileron_yaw | heldout | elevator | 0.1925666461 | 0.3786706329 | 0.1661870446 | 12.42552277 | 9.797630992 | 6.530439619 | 0.524004105`
- `C7_c6_plus_alpha_regime_aileron_yaw | heldout | rudder | 0.3071569707 | 0.3646306841 | 0.1442407244 | 17.12004734 | 8.018639192 | 7.781466469 | 0.5784324581`
- `C8_c7_plus_alpha_regime_aileron_side_force | all | all | 0.2573454958 | 0.5238213432 | 0.256961416 | 16.08775233 | 9.875080555 | 8.900595116 | 0.2053384332`
- `C8_c7_plus_alpha_regime_aileron_side_force | all | aileron | 0.332418708 | 0.7289195808 | 0.4183069604 | 17.62979659 | 12.92601528 | 10.9535467 | 0.2070870192`
- `C8_c7_plus_alpha_regime_aileron_side_force | all | elevator | 0.1818802774 | 0.4149092356 | 0.1925118908 | 13.05164984 | 9.576150987 | 8.09795549 | 0.2098580928`
- `C8_c7_plus_alpha_regime_aileron_side_force | all | rudder | 0.2551420117 | 0.4222212417 | 0.1561723783 | 17.502877 | 7.065319119 | 7.60104883 | 0.1990701877`
- `C8_c7_plus_alpha_regime_aileron_side_force | heldout | all | 0.2769405069 | 0.4493870682 | 0.2387718148 | 15.70700956 | 10.54183718 | 8.589007827 | 0.4358836003`
- `C8_c7_plus_alpha_regime_aileron_side_force | heldout | aileron | 0.3310979038 | 0.6048598876 | 0.4058876755 | 17.57545857 | 13.80924136 | 11.45511739 | 0.2052142377`
- `C8_c7_plus_alpha_regime_aileron_side_force | heldout | elevator | 0.1925666461 | 0.3786706329 | 0.1661870446 | 12.42552277 | 9.797630992 | 6.530439619 | 0.524004105`
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
- `C0_frozen_neutral | elevator | transition | 0.2 | 1 | 0.02616005092 | 0.1928126069 | 0.0503834155 | 21.71903957 | 5.019178153 | 12.34559374 | `
- `C0_frozen_neutral | elevator | transition | 0.4 | 1 | 0.2434969252 | 0.04273636585 | 0.01852382454 | 14.25266083 | 17.39888116 | 1.627061226 | `
- `C0_frozen_neutral | elevator | transition | 0.6 | 1 | 0.02634436111 | 0.5826907011 | 0.01657480623 | 14.141615 | 14.55429935 | 4.835637465 | `
- `C0_frozen_neutral | elevator | transition | 0.8 | 1 | 0.1617455638 | 0.5662567311 | 0.1417127234 | 14.08020336 | 32.96028875 | 3.302811455 | `
- `C0_frozen_neutral | elevator | transition | 1 | 1 | 0.1149084632 | 0.3775022995 | 0.4788332301 | 0.4896157481 | 31.84649183 | 1.367105158 | `
- `C0_frozen_neutral | elevator | post_stall | 0.2 | 1 | 0.03123621709 | 0.5740109866 | 0.01114070028 | 13.16231187 | 7.797989761 | 10.79375145 | `
- `C0_frozen_neutral | elevator | post_stall | 0.4 | 1 | 0.3930195856 | 0.1869173608 | 0.1465531443 | 19.30880336 | 26.93605501 | 4.905250713 | `
- `C0_frozen_neutral | elevator | post_stall | 0.6 | 1 | 0.04684611321 | 0.3693532669 | 0.07329729753 | 18.4304999 | 16.76243835 | 10.56274582 | `
- `C0_frozen_neutral | elevator | post_stall | 0.8 | 1 | 0.04736135784 | 0.6608680434 | 0.1377253653 | 12.3336916 | 23.3523047 | 9.310930192 | `
- `C0_frozen_neutral | elevator | post_stall | 1 | 1 | 0.05962226771 | 0.2551263902 | 0.0008664010952 | 2.977250658 | 22.71064372 | 8.841675074 | `
- `C0_frozen_neutral | rudder | transition | 0.2 | 2 | 0.377380812 | 0.1265475992 | 0.3734611931 | 16.96988245 | 9.324813503 | 7.770061952 | 0.8218151753`
- `C0_frozen_neutral | rudder | transition | 0.4 | 2 | 0.1352121599 | 0.3562545114 | 0.08019368549 | 19.52697663 | 12.17784025 | 6.426630687 | 0.5711831517`
- `C0_frozen_neutral | rudder | transition | 0.6 | 2 | 0.134123656 | 0.3831172802 | 0.02384417233 | 12.43261012 | 1.886094046 | 8.349837733 | 0.2906263145`
- `C0_frozen_neutral | rudder | transition | 0.8 | 2 | 0.2057581688 | 0.7373711728 | 0.1490897537 | 8.530993454 | 16.05666527 | 26.47619571 | 0.5931191604`
- `C0_frozen_neutral | rudder | transition | 1 | 2 | 0.3451218638 | 0.8880860216 | 0.20273097 | 12.39073498 | 6.050390518 | 24.0044814 | 0.5812632339`
- `P1_pairwise_aileron_gain | aileron | transition | 0.2 | 2 | 0.4521926741 | 0.2484927587 | 0.2389557422 | 18.41434248 | 6.268381562 | 7.292610213 | 0.2180067227`
- `P1_pairwise_aileron_gain | aileron | transition | 0.4 | 2 | 0.3851782378 | 0.5102412017 | 0.4639346403 | 14.7836134 | 10.50873536 | 5.525461269 | 0.2786385135`
- `P1_pairwise_aileron_gain | aileron | transition | 0.6 | 1 | 0.9021318069 | 0.003979160181 | 0.9091373431 | 28.81850483 | 15.11453722 | 8.435870685 | `
- `P1_pairwise_aileron_gain | aileron | transition | 0.8 | 2 | 0.08155631527 | 0.1226879303 | 0.03282269049 | 23.02021045 | 7.415468486 | 33.96663136 | 0.1949656034`
- `P1_pairwise_aileron_gain | aileron | post_stall | 0.6 | 1 | 0.0001349562666 | 0.4608852858 | 0.1098621167 | 6.461510564 | 3.009308877 | 52.06622847 | `
- `P1_pairwise_aileron_gain | aileron | post_stall | 1 | 2 | 0.4854236472 | 0.8018293942 | 0.1732200711 | 24.37230024 | 21.09802441 | 59.67812097 | 0.3060345738`
- `P1_pairwise_aileron_gain | elevator | transition | 0.2 | 1 | 0.02616005092 | 0.1928126069 | 0.0503834155 | 21.71903957 | 5.019178153 | 12.34559374 | `
- `P1_pairwise_aileron_gain | elevator | transition | 0.4 | 1 | 0.2434969252 | 0.04273636585 | 0.01852382454 | 14.25266083 | 17.39888116 | 1.627061226 | `
- `P1_pairwise_aileron_gain | elevator | transition | 0.6 | 1 | 0.02634436111 | 0.5826907011 | 0.01657480623 | 14.141615 | 14.55429935 | 4.835637465 | `
- `P1_pairwise_aileron_gain | elevator | transition | 0.8 | 1 | 0.1617455638 | 0.5662567311 | 0.1417127234 | 14.08020336 | 32.96028875 | 3.302811455 | `
- `P1_pairwise_aileron_gain | elevator | transition | 1 | 1 | 0.1149084632 | 0.3775022995 | 0.4788332301 | 0.4896157481 | 31.84649183 | 1.367105158 | `
- `P1_pairwise_aileron_gain | elevator | post_stall | 0.2 | 1 | 0.03123621709 | 0.5740109866 | 0.01114070028 | 13.16231187 | 7.797989761 | 10.79375145 | `
- `P1_pairwise_aileron_gain | elevator | post_stall | 0.4 | 1 | 0.3930195856 | 0.1869173608 | 0.1465531443 | 19.30880336 | 26.93605501 | 4.905250713 | `
- `P1_pairwise_aileron_gain | elevator | post_stall | 0.6 | 1 | 0.04684611321 | 0.3693532669 | 0.07329729753 | 18.4304999 | 16.76243835 | 10.56274582 | `
- `P1_pairwise_aileron_gain | elevator | post_stall | 0.8 | 1 | 0.04736135784 | 0.6608680434 | 0.1377253653 | 12.3336916 | 23.3523047 | 9.310930192 | `
- `P1_pairwise_aileron_gain | elevator | post_stall | 1 | 1 | 0.05962226771 | 0.2551263902 | 0.0008664010952 | 2.977250658 | 22.71064372 | 8.841675074 | `
- `P1_pairwise_aileron_gain | rudder | transition | 0.2 | 2 | 0.377380812 | 0.1265475992 | 0.3734611931 | 16.96988245 | 9.324813503 | 7.770061952 | 0.8218151753`
- `P1_pairwise_aileron_gain | rudder | transition | 0.4 | 2 | 0.1352121599 | 0.3562545114 | 0.08019368549 | 19.52697663 | 12.17784025 | 6.426630687 | 0.5711831517`
- `P1_pairwise_aileron_gain | rudder | transition | 0.6 | 2 | 0.134123656 | 0.3831172802 | 0.02384417233 | 12.43261012 | 1.886094046 | 8.349837733 | 0.2906263145`
- `P1_pairwise_aileron_gain | rudder | transition | 0.8 | 2 | 0.2057581688 | 0.7373711728 | 0.1490897537 | 8.530993454 | 16.05666527 | 26.47619571 | 0.5931191604`
- `P1_pairwise_aileron_gain | rudder | transition | 1 | 2 | 0.3451218638 | 0.8880860216 | 0.20273097 | 12.39073498 | 6.050390518 | 24.0044814 | 0.5812632339`
- `P2_pairwise_elevator_gain | aileron | transition | 0.2 | 2 | 0.4504285959 | 0.2530196281 | 0.2451856179 | 19.76557821 | 6.103107888 | 7.720275058 | 0.1661385645`
- `P2_pairwise_elevator_gain | aileron | transition | 0.4 | 2 | 0.3759545656 | 0.5306732028 | 0.4728628765 | 18.73200363 | 11.49486243 | 6.540730445 | 0.1738168885`
- `P2_pairwise_elevator_gain | aileron | transition | 0.6 | 1 | 0.8681745221 | 0.01671321325 | 0.9611264127 | 39.10448952 | 17.06961034 | 7.456566527 | `
- `P2_pairwise_elevator_gain | aileron | transition | 0.8 | 2 | 0.1166013734 | 0.1255892131 | 0.02820657031 | 24.04501034 | 7.753203696 | 39.72699223 | 0.1253302209`
- `P2_pairwise_elevator_gain | aileron | post_stall | 0.6 | 1 | 0.04340959713 | 0.417080028 | 0.1162668355 | 5.920577236 | 1.641215174 | 57.88166922 | `
- `P2_pairwise_elevator_gain | aileron | post_stall | 1 | 2 | 0.5541461733 | 0.7841212898 | 0.2427651567 | 35.41218707 | 22.63026016 | 63.81557994 | 0.4574340949`
- `P2_pairwise_elevator_gain | elevator | transition | 0.2 | 1 | 0.01402573304 | 0.19133176 | 0.04578089396 | 19.96843426 | 4.240486457 | 12.39802567 | `
- `P2_pairwise_elevator_gain | elevator | transition | 0.4 | 1 | 0.3672041126 | 0.009000086612 | 0.4774804244 | 7.114964723 | 15.28268879 | 2.908629733 | `
- `P2_pairwise_elevator_gain | elevator | transition | 0.6 | 1 | 0.03208343794 | 0.5832631816 | 0.01754256423 | 15.34332068 | 16.08536011 | 4.700439214 | `
- `P2_pairwise_elevator_gain | elevator | transition | 0.8 | 1 | 0.1446098932 | 0.5629774211 | 0.1633815251 | 19.68317442 | 37.04539359 | 3.204035857 | `
- `P2_pairwise_elevator_gain | elevator | transition | 1 | 1 | 0.07186257325 | 0.385679041 | 0.5258993663 | 12.15008926 | 38.19967688 | 1.240055833 | `
- `P2_pairwise_elevator_gain | elevator | post_stall | 0.2 | 1 | 0.01773453385 | 0.5761940706 | 0.02706914463 | 13.20135716 | 5.319931413 | 10.49073022 | `
- `P2_pairwise_elevator_gain | elevator | post_stall | 0.4 | 1 | 0.4246203548 | 0.1835864762 | 0.1257413886 | 18.70821398 | 22.19424514 | 3.674044247 | `
- `P2_pairwise_elevator_gain | elevator | post_stall | 0.6 | 1 | 0.06969293014 | 0.3750008684 | 0.104994959 | 19.15769585 | 8.626862613 | 9.321354903 | `
- `P2_pairwise_elevator_gain | elevator | post_stall | 0.8 | 1 | 0.08071834129 | 0.6695653406 | 0.1546871414 | 12.1942362 | 13.89496657 | 9.549386826 | `
- `P2_pairwise_elevator_gain | elevator | post_stall | 1 | 1 | 0.1741977511 | 0.256128151 | 0.006350488841 | 2.097522785 | 12.9261284 | 7.511658516 | `
- `P2_pairwise_elevator_gain | rudder | transition | 0.2 | 2 | 0.377380812 | 0.1265475992 | 0.3734611931 | 16.96988245 | 9.324813503 | 7.770061952 | 0.8218151753`
- `P2_pairwise_elevator_gain | rudder | transition | 0.4 | 2 | 0.1352121599 | 0.3562545114 | 0.08019368549 | 19.52697663 | 12.17784025 | 6.426630687 | 0.5711831517`
- `P2_pairwise_elevator_gain | rudder | transition | 0.6 | 2 | 0.134123656 | 0.3831172802 | 0.02384417233 | 12.43261012 | 1.886094046 | 8.349837733 | 0.2906263145`
- `P2_pairwise_elevator_gain | rudder | transition | 0.8 | 2 | 0.2057581688 | 0.7373711728 | 0.1490897537 | 8.530993454 | 16.05666527 | 26.47619571 | 0.5931191604`
- `P2_pairwise_elevator_gain | rudder | transition | 1 | 2 | 0.3451218638 | 0.8880860216 | 0.20273097 | 12.39073498 | 6.050390518 | 24.0044814 | 0.5812632339`
- `P3_pairwise_rudder_gain | aileron | transition | 0.2 | 2 | 0.4504285959 | 0.2530196281 | 0.2451856179 | 19.76557821 | 6.103107888 | 7.720275058 | 0.1661385645`
- `P3_pairwise_rudder_gain | aileron | transition | 0.4 | 2 | 0.3759545656 | 0.5306732028 | 0.4728628765 | 18.73200363 | 11.49486243 | 6.540730445 | 0.1738168885`
- `P3_pairwise_rudder_gain | aileron | transition | 0.6 | 1 | 0.8681745221 | 0.01671321325 | 0.9611264127 | 39.10448952 | 17.06961034 | 7.456566527 | `
- `P3_pairwise_rudder_gain | aileron | transition | 0.8 | 2 | 0.1166013734 | 0.1255892131 | 0.02820657031 | 24.04501034 | 7.753203696 | 39.72699223 | 0.1253302209`
- `P3_pairwise_rudder_gain | aileron | post_stall | 0.6 | 1 | 0.04340959713 | 0.417080028 | 0.1162668355 | 5.920577236 | 1.641215174 | 57.88166922 | `
- `P3_pairwise_rudder_gain | aileron | post_stall | 1 | 2 | 0.5541461733 | 0.7841212898 | 0.2427651567 | 35.41218707 | 22.63026016 | 63.81557994 | 0.4574340949`
- `P3_pairwise_rudder_gain | elevator | transition | 0.2 | 1 | 0.02616005092 | 0.1928126069 | 0.0503834155 | 21.71903957 | 5.019178153 | 12.34559374 | `
- `P3_pairwise_rudder_gain | elevator | transition | 0.4 | 1 | 0.2434969252 | 0.04273636585 | 0.01852382454 | 14.25266083 | 17.39888116 | 1.627061226 | `
- `P3_pairwise_rudder_gain | elevator | transition | 0.6 | 1 | 0.02634436111 | 0.5826907011 | 0.01657480623 | 14.141615 | 14.55429935 | 4.835637465 | `
- `P3_pairwise_rudder_gain | elevator | transition | 0.8 | 1 | 0.1617455638 | 0.5662567311 | 0.1417127234 | 14.08020336 | 32.96028875 | 3.302811455 | `
- `P3_pairwise_rudder_gain | elevator | transition | 1 | 1 | 0.1149084632 | 0.3775022995 | 0.4788332301 | 0.4896157481 | 31.84649183 | 1.367105158 | `
- `P3_pairwise_rudder_gain | elevator | post_stall | 0.2 | 1 | 0.03123621709 | 0.5740109866 | 0.01114070028 | 13.16231187 | 7.797989761 | 10.79375145 | `
- `P3_pairwise_rudder_gain | elevator | post_stall | 0.4 | 1 | 0.3930195856 | 0.1869173608 | 0.1465531443 | 19.30880336 | 26.93605501 | 4.905250713 | `
- `P3_pairwise_rudder_gain | elevator | post_stall | 0.6 | 1 | 0.04684611321 | 0.3693532669 | 0.07329729753 | 18.4304999 | 16.76243835 | 10.56274582 | `
- `P3_pairwise_rudder_gain | elevator | post_stall | 0.8 | 1 | 0.04736135784 | 0.6608680434 | 0.1377253653 | 12.3336916 | 23.3523047 | 9.310930192 | `
- `P3_pairwise_rudder_gain | elevator | post_stall | 1 | 1 | 0.05962226771 | 0.2551263902 | 0.0008664010952 | 2.977250658 | 22.71064372 | 8.841675074 | `
- `P3_pairwise_rudder_gain | rudder | transition | 0.2 | 2 | 0.3876794609 | 0.08831290014 | 0.3692992315 | 17.48391081 | 8.833823664 | 4.098316232 | 0.8897885423`
- `P3_pairwise_rudder_gain | rudder | transition | 0.4 | 2 | 0.1431332587 | 0.3977902115 | 0.08010606876 | 21.37761063 | 12.39510702 | 8.046310735 | 0.3525179522`
- `P3_pairwise_rudder_gain | rudder | transition | 0.6 | 2 | 0.1155601502 | 0.4305366389 | 0.02136143122 | 13.85694579 | 2.549885773 | 3.108922394 | 0.00263067733`
- `P3_pairwise_rudder_gain | rudder | transition | 0.8 | 2 | 0.2724617327 | 0.6730589813 | 0.1491018566 | 12.92552303 | 13.06331601 | 14.94804637 | 0.05076885307`
- `P3_pairwise_rudder_gain | rudder | transition | 1 | 2 | 0.4506440495 | 0.851097248 | 0.1873784598 | 16.51240149 | 4.010240651 | 11.6786884 | 0.009618642608`
- `P4_pairwise_all_surface_gains | aileron | transition | 0.2 | 2 | 0.4521926741 | 0.2484927587 | 0.2389557422 | 18.41434248 | 6.268381562 | 7.292610213 | 0.2180067227`
- `P4_pairwise_all_surface_gains | aileron | transition | 0.4 | 2 | 0.3851782378 | 0.5102412017 | 0.4639346403 | 14.7836134 | 10.50873536 | 5.525461269 | 0.2786385135`
- `P4_pairwise_all_surface_gains | aileron | transition | 0.6 | 1 | 0.9021318069 | 0.003979160181 | 0.9091373431 | 28.81850483 | 15.11453722 | 8.435870685 | `
- `P4_pairwise_all_surface_gains | aileron | transition | 0.8 | 2 | 0.08155631527 | 0.1226879303 | 0.03282269049 | 23.02021045 | 7.415468486 | 33.96663136 | 0.1949656034`
- `P4_pairwise_all_surface_gains | aileron | post_stall | 0.6 | 1 | 0.0001349562666 | 0.4608852858 | 0.1098621167 | 6.461510564 | 3.009308877 | 52.06622847 | `
- `P4_pairwise_all_surface_gains | aileron | post_stall | 1 | 2 | 0.4854236472 | 0.8018293942 | 0.1732200711 | 24.37230024 | 21.09802441 | 59.67812097 | 0.3060345738`
- `P4_pairwise_all_surface_gains | elevator | transition | 0.2 | 1 | 0.01402573304 | 0.19133176 | 0.04578089396 | 19.96843426 | 4.240486457 | 12.39802567 | `
- `P4_pairwise_all_surface_gains | elevator | transition | 0.4 | 1 | 0.3672041126 | 0.009000086612 | 0.4774804244 | 7.114964723 | 15.28268879 | 2.908629733 | `
- `P4_pairwise_all_surface_gains | elevator | transition | 0.6 | 1 | 0.03208343794 | 0.5832631816 | 0.01754256423 | 15.34332068 | 16.08536011 | 4.700439214 | `
- `P4_pairwise_all_surface_gains | elevator | transition | 0.8 | 1 | 0.1446098932 | 0.5629774211 | 0.1633815251 | 19.68317442 | 37.04539359 | 3.204035857 | `
- `P4_pairwise_all_surface_gains | elevator | transition | 1 | 1 | 0.07186257325 | 0.385679041 | 0.5258993663 | 12.15008926 | 38.19967688 | 1.240055833 | `
- `P4_pairwise_all_surface_gains | elevator | post_stall | 0.2 | 1 | 0.01773453385 | 0.5761940706 | 0.02706914463 | 13.20135716 | 5.319931413 | 10.49073022 | `
- `P4_pairwise_all_surface_gains | elevator | post_stall | 0.4 | 1 | 0.4246203548 | 0.1835864762 | 0.1257413886 | 18.70821398 | 22.19424514 | 3.674044247 | `
- `P4_pairwise_all_surface_gains | elevator | post_stall | 0.6 | 1 | 0.06969293014 | 0.3750008684 | 0.104994959 | 19.15769585 | 8.626862613 | 9.321354903 | `
- `P4_pairwise_all_surface_gains | elevator | post_stall | 0.8 | 1 | 0.08071834129 | 0.6695653406 | 0.1546871414 | 12.1942362 | 13.89496657 | 9.549386826 | `
- `P4_pairwise_all_surface_gains | elevator | post_stall | 1 | 1 | 0.1741977511 | 0.256128151 | 0.006350488841 | 2.097522785 | 12.9261284 | 7.511658516 | `
- `P4_pairwise_all_surface_gains | rudder | transition | 0.2 | 2 | 0.3876794609 | 0.08831290014 | 0.3692992315 | 17.48391081 | 8.833823664 | 4.098316232 | 0.8897885423`
- `P4_pairwise_all_surface_gains | rudder | transition | 0.4 | 2 | 0.1431332587 | 0.3977902115 | 0.08010606876 | 21.37761063 | 12.39510702 | 8.046310735 | 0.3525179522`
- `P4_pairwise_all_surface_gains | rudder | transition | 0.6 | 2 | 0.1155601502 | 0.4305366389 | 0.02136143122 | 13.85694579 | 2.549885773 | 3.108922394 | 0.00263067733`
- `P4_pairwise_all_surface_gains | rudder | transition | 0.8 | 2 | 0.2724617327 | 0.6730589813 | 0.1491018566 | 12.92552303 | 13.06331601 | 14.94804637 | 0.05076885307`
- `P4_pairwise_all_surface_gains | rudder | transition | 1 | 2 | 0.4506440495 | 0.851097248 | 0.1873784598 | 16.51240149 | 4.010240651 | 11.6786884 | 0.009618642608`
- `C1_primary_moment_derivatives | aileron | transition | 0.2 | 2 | 0.4499052779 | 0.2385401049 | 0.2368853984 | 20.28372936 | 6.598700005 | 6.685003006 | 0.2948156863`
- `C1_primary_moment_derivatives | aileron | transition | 0.4 | 2 | 0.3738235596 | 0.5000816506 | 0.4664340154 | 11.41963098 | 10.74652327 | 5.702543456 | 0.3357760766`
- `C1_primary_moment_derivatives | aileron | transition | 0.6 | 1 | 0.8644254084 | 0.04103804474 | 0.935824806 | 28.24866505 | 15.18903256 | 9.92364156 | `
- `C1_primary_moment_derivatives | aileron | transition | 0.8 | 2 | 0.106013837 | 0.1095272163 | 0.02243142495 | 24.83714723 | 5.34580526 | 34.10127681 | 0.3311265956`
- `C1_primary_moment_derivatives | aileron | post_stall | 0.6 | 1 | 0.02258546157 | 0.4559191382 | 0.117964632 | 17.173813 | 5.515410421 | 51.58472659 | `
- `C1_primary_moment_derivatives | aileron | post_stall | 1 | 2 | 0.5398570444 | 0.8202083576 | 0.280073414 | 31.55088453 | 27.59926382 | 59.01953457 | 0.159993069`
- `C1_primary_moment_derivatives | elevator | transition | 0.2 | 1 | 0.02499219899 | 0.1928362654 | 0.05014879186 | 21.53861812 | 4.95943507 | 12.34228534 | `
- `C1_primary_moment_derivatives | elevator | transition | 0.4 | 1 | 0.2361826956 | 0.04230328817 | 0.02161411611 | 14.44197951 | 16.98640483 | 1.831542686 | `
- `C1_primary_moment_derivatives | elevator | transition | 0.6 | 1 | 0.02744650499 | 0.5826686989 | 0.01562473032 | 14.22688145 | 14.79328696 | 4.837853854 | `
- `C1_primary_moment_derivatives | elevator | transition | 0.8 | 1 | 0.1581611061 | 0.5657575124 | 0.1480813487 | 14.17806268 | 33.68449132 | 3.330898409 | `
- `C1_primary_moment_derivatives | elevator | transition | 1 | 1 | 0.1077862856 | 0.3788407009 | 0.4919547134 | 0.6151676763 | 33.2375546 | 1.191936063 | `
- `C1_primary_moment_derivatives | elevator | post_stall | 0.2 | 1 | 0.03014140975 | 0.5741589811 | 0.01229867579 | 13.16269015 | 7.618458085 | 10.77278312 | `
- `C1_primary_moment_derivatives | elevator | post_stall | 0.4 | 1 | 0.3982022069 | 0.1865614283 | 0.144354838 | 19.28957746 | 26.57971501 | 4.807611615 | `
- `C1_primary_moment_derivatives | elevator | post_stall | 0.6 | 1 | 0.03898484956 | 0.369650158 | 0.0746232801 | 18.47870372 | 16.25874172 | 10.4812783 | `
- `C1_primary_moment_derivatives | elevator | post_stall | 0.8 | 1 | 0.03903808243 | 0.6614349458 | 0.1380524955 | 12.32586866 | 22.7902596 | 9.32528598 | `
- `C1_primary_moment_derivatives | elevator | post_stall | 1 | 1 | 0.06722188162 | 0.255157338 | 0.0006338383452 | 2.917931802 | 22.12471058 | 8.752808779 | `
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
- `C2_c1_plus_aileron_adverse_yaw | elevator | transition | 0.2 | 1 | 0.02499219899 | 0.1928362654 | 0.05014879186 | 21.53861812 | 4.95943507 | 12.34228534 | `
- `C2_c1_plus_aileron_adverse_yaw | elevator | transition | 0.4 | 1 | 0.2361826956 | 0.04230328817 | 0.02161411611 | 14.44197951 | 16.98640483 | 1.831542686 | `
- `C2_c1_plus_aileron_adverse_yaw | elevator | transition | 0.6 | 1 | 0.02744650499 | 0.5826686989 | 0.01562473032 | 14.22688145 | 14.79328696 | 4.837853854 | `
- `C2_c1_plus_aileron_adverse_yaw | elevator | transition | 0.8 | 1 | 0.1581611061 | 0.5657575124 | 0.1480813487 | 14.17806268 | 33.68449132 | 3.330898409 | `
- `C2_c1_plus_aileron_adverse_yaw | elevator | transition | 1 | 1 | 0.1077862856 | 0.3788407009 | 0.4919547134 | 0.6151676763 | 33.2375546 | 1.191936063 | `
- `C2_c1_plus_aileron_adverse_yaw | elevator | post_stall | 0.2 | 1 | 0.03014140975 | 0.5741589811 | 0.01229867579 | 13.16269015 | 7.618458085 | 10.77278312 | `
- `C2_c1_plus_aileron_adverse_yaw | elevator | post_stall | 0.4 | 1 | 0.3982022069 | 0.1865614283 | 0.144354838 | 19.28957746 | 26.57971501 | 4.807611615 | `
- `C2_c1_plus_aileron_adverse_yaw | elevator | post_stall | 0.6 | 1 | 0.03898484956 | 0.369650158 | 0.0746232801 | 18.47870372 | 16.25874172 | 10.4812783 | `
- `C2_c1_plus_aileron_adverse_yaw | elevator | post_stall | 0.8 | 1 | 0.03903808243 | 0.6614349458 | 0.1380524955 | 12.32586866 | 22.7902596 | 9.32528598 | `
- `C2_c1_plus_aileron_adverse_yaw | elevator | post_stall | 1 | 1 | 0.06722188162 | 0.255157338 | 0.0006338383452 | 2.917931802 | 22.12471058 | 8.752808779 | `
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
- `C3_c1_plus_rudder_roll | elevator | transition | 0.2 | 1 | 0.02499219899 | 0.1928362654 | 0.05014879186 | 21.53861812 | 4.95943507 | 12.34228534 | `
- `C3_c1_plus_rudder_roll | elevator | transition | 0.4 | 1 | 0.2361826956 | 0.04230328817 | 0.02161411611 | 14.44197951 | 16.98640483 | 1.831542686 | `
- `C3_c1_plus_rudder_roll | elevator | transition | 0.6 | 1 | 0.02744650499 | 0.5826686989 | 0.01562473032 | 14.22688145 | 14.79328696 | 4.837853854 | `
- `C3_c1_plus_rudder_roll | elevator | transition | 0.8 | 1 | 0.1581611061 | 0.5657575124 | 0.1480813487 | 14.17806268 | 33.68449132 | 3.330898409 | `
- `C3_c1_plus_rudder_roll | elevator | transition | 1 | 1 | 0.1077862856 | 0.3788407009 | 0.4919547134 | 0.6151676763 | 33.2375546 | 1.191936063 | `
- `C3_c1_plus_rudder_roll | elevator | post_stall | 0.2 | 1 | 0.03014140975 | 0.5741589811 | 0.01229867579 | 13.16269015 | 7.618458085 | 10.77278312 | `
- `C3_c1_plus_rudder_roll | elevator | post_stall | 0.4 | 1 | 0.3982022069 | 0.1865614283 | 0.144354838 | 19.28957746 | 26.57971501 | 4.807611615 | `
- `C3_c1_plus_rudder_roll | elevator | post_stall | 0.6 | 1 | 0.03898484956 | 0.369650158 | 0.0746232801 | 18.47870372 | 16.25874172 | 10.4812783 | `
- `C3_c1_plus_rudder_roll | elevator | post_stall | 0.8 | 1 | 0.03903808243 | 0.6614349458 | 0.1380524955 | 12.32586866 | 22.7902596 | 9.32528598 | `
- `C3_c1_plus_rudder_roll | elevator | post_stall | 1 | 1 | 0.06722188162 | 0.255157338 | 0.0006338383452 | 2.917931802 | 22.12471058 | 8.752808779 | `
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
- `C4_c1_plus_surface_side_force | elevator | transition | 0.2 | 1 | 0.02499219899 | 0.1928362654 | 0.05014879186 | 21.53861812 | 4.95943507 | 12.34228534 | `
- `C4_c1_plus_surface_side_force | elevator | transition | 0.4 | 1 | 0.2361826956 | 0.04230328817 | 0.02161411611 | 14.44197951 | 16.98640483 | 1.831542686 | `
- `C4_c1_plus_surface_side_force | elevator | transition | 0.6 | 1 | 0.02744650499 | 0.5826686989 | 0.01562473032 | 14.22688145 | 14.79328696 | 4.837853854 | `
- `C4_c1_plus_surface_side_force | elevator | transition | 0.8 | 1 | 0.1581611061 | 0.5657575124 | 0.1480813487 | 14.17806268 | 33.68449132 | 3.330898409 | `
- `C4_c1_plus_surface_side_force | elevator | transition | 1 | 1 | 0.1077862856 | 0.3788407009 | 0.4919547134 | 0.6151676763 | 33.2375546 | 1.191936063 | `
- `C4_c1_plus_surface_side_force | elevator | post_stall | 0.2 | 1 | 0.03014140975 | 0.5741589811 | 0.01229867579 | 13.16269015 | 7.618458085 | 10.77278312 | `
- `C4_c1_plus_surface_side_force | elevator | post_stall | 0.4 | 1 | 0.3982022069 | 0.1865614283 | 0.144354838 | 19.28957746 | 26.57971501 | 4.807611615 | `
- `C4_c1_plus_surface_side_force | elevator | post_stall | 0.6 | 1 | 0.03898484956 | 0.369650158 | 0.0746232801 | 18.47870372 | 16.25874172 | 10.4812783 | `
- `C4_c1_plus_surface_side_force | elevator | post_stall | 0.8 | 1 | 0.03903808243 | 0.6614349458 | 0.1380524955 | 12.32586866 | 22.7902596 | 9.32528598 | `
- `C4_c1_plus_surface_side_force | elevator | post_stall | 1 | 1 | 0.06722188162 | 0.255157338 | 0.0006338383452 | 2.917931802 | 22.12471058 | 8.752808779 | `
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
- `C5_c2_plus_surface_side_force | elevator | transition | 0.2 | 1 | 0.02499219899 | 0.1928362654 | 0.05014879186 | 21.53861812 | 4.95943507 | 12.34228534 | `
- `C5_c2_plus_surface_side_force | elevator | transition | 0.4 | 1 | 0.2361826956 | 0.04230328817 | 0.02161411611 | 14.44197951 | 16.98640483 | 1.831542686 | `
- `C5_c2_plus_surface_side_force | elevator | transition | 0.6 | 1 | 0.02744650499 | 0.5826686989 | 0.01562473032 | 14.22688145 | 14.79328696 | 4.837853854 | `
- `C5_c2_plus_surface_side_force | elevator | transition | 0.8 | 1 | 0.1581611061 | 0.5657575124 | 0.1480813487 | 14.17806268 | 33.68449132 | 3.330898409 | `
- `C5_c2_plus_surface_side_force | elevator | transition | 1 | 1 | 0.1077862856 | 0.3788407009 | 0.4919547134 | 0.6151676763 | 33.2375546 | 1.191936063 | `
- `C5_c2_plus_surface_side_force | elevator | post_stall | 0.2 | 1 | 0.03014140975 | 0.5741589811 | 0.01229867579 | 13.16269015 | 7.618458085 | 10.77278312 | `
- `C5_c2_plus_surface_side_force | elevator | post_stall | 0.4 | 1 | 0.3982022069 | 0.1865614283 | 0.144354838 | 19.28957746 | 26.57971501 | 4.807611615 | `
- `C5_c2_plus_surface_side_force | elevator | post_stall | 0.6 | 1 | 0.03898484956 | 0.369650158 | 0.0746232801 | 18.47870372 | 16.25874172 | 10.4812783 | `
- `C5_c2_plus_surface_side_force | elevator | post_stall | 0.8 | 1 | 0.03903808243 | 0.6614349458 | 0.1380524955 | 12.32586866 | 22.7902596 | 9.32528598 | `
- `C5_c2_plus_surface_side_force | elevator | post_stall | 1 | 1 | 0.06722188162 | 0.255157338 | 0.0006338383452 | 2.917931802 | 22.12471058 | 8.752808779 | `
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
- `C6_alpha_regime_primary_derivatives | elevator | transition | 0.2 | 1 | 0.03514568241 | 0.1922533016 | 0.05134740803 | 23.19565304 | 5.510792551 | 12.41237255 | `
- `C6_alpha_regime_primary_derivatives | elevator | transition | 0.4 | 1 | 0.3345009861 | 0.006702783071 | 0.684198053 | 7.44388449 | 25.95531786 | 3.992255983 | `
- `C6_alpha_regime_primary_derivatives | elevator | transition | 0.6 | 1 | 0.01759335793 | 0.5835096014 | 0.03446190055 | 13.62501317 | 9.318694557 | 4.572206256 | `
- `C6_alpha_regime_primary_derivatives | elevator | transition | 0.8 | 1 | 0.2033673874 | 0.5724988678 | 0.01603495607 | 13.5827088 | 14.80083128 | 2.772055922 | `
- `C6_alpha_regime_primary_derivatives | elevator | transition | 1 | 1 | 0.1546895227 | 0.3430166752 | 0.1804882375 | 1.603723552 | 4.698571933 | 6.575000553 | `
- `C6_alpha_regime_primary_derivatives | elevator | post_stall | 0.2 | 1 | 0.0141268322 | 0.5801979004 | 0.05803946557 | 13.7390615 | 1.237664591 | 9.888670023 | `
- `C6_alpha_regime_primary_derivatives | elevator | post_stall | 0.4 | 1 | 0.4078806164 | 0.1832099995 | 0.1197822161 | 18.39421087 | 16.59185538 | 2.492769222 | `
- `C6_alpha_regime_primary_derivatives | elevator | post_stall | 0.6 | 1 | 0.1959485261 | 0.3863741154 | 0.1966535344 | 19.69356216 | 8.027764405 | 7.226163884 | `
- `C6_alpha_regime_primary_derivatives | elevator | post_stall | 0.8 | 1 | 0.1834492625 | 0.6749124759 | 0.2121687841 | 11.9499891 | 2.729980863 | 10.04162593 | `
- `C6_alpha_regime_primary_derivatives | elevator | post_stall | 1 | 1 | 0.378964287 | 0.264030609 | 0.1086958907 | 1.027421036 | 9.104836485 | 5.331275873 | `
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
- `C7_c6_plus_alpha_regime_aileron_yaw | elevator | transition | 0.2 | 1 | 0.03514568241 | 0.1922533016 | 0.05134740803 | 23.19565304 | 5.510792551 | 12.41237255 | `
- `C7_c6_plus_alpha_regime_aileron_yaw | elevator | transition | 0.4 | 1 | 0.3345009861 | 0.006702783071 | 0.684198053 | 7.44388449 | 25.95531786 | 3.992255983 | `
- `C7_c6_plus_alpha_regime_aileron_yaw | elevator | transition | 0.6 | 1 | 0.01759335793 | 0.5835096014 | 0.03446190055 | 13.62501317 | 9.318694557 | 4.572206256 | `
- `C7_c6_plus_alpha_regime_aileron_yaw | elevator | transition | 0.8 | 1 | 0.2033673874 | 0.5724988678 | 0.01603495607 | 13.5827088 | 14.80083128 | 2.772055922 | `
- `C7_c6_plus_alpha_regime_aileron_yaw | elevator | transition | 1 | 1 | 0.1546895227 | 0.3430166752 | 0.1804882375 | 1.603723552 | 4.698571933 | 6.575000553 | `
- `C7_c6_plus_alpha_regime_aileron_yaw | elevator | post_stall | 0.2 | 1 | 0.0141268322 | 0.5801979004 | 0.05803946557 | 13.7390615 | 1.237664591 | 9.888670023 | `
- `C7_c6_plus_alpha_regime_aileron_yaw | elevator | post_stall | 0.4 | 1 | 0.4078806164 | 0.1832099995 | 0.1197822161 | 18.39421087 | 16.59185538 | 2.492769222 | `
- `C7_c6_plus_alpha_regime_aileron_yaw | elevator | post_stall | 0.6 | 1 | 0.1959485261 | 0.3863741154 | 0.1966535344 | 19.69356216 | 8.027764405 | 7.226163884 | `
- `C7_c6_plus_alpha_regime_aileron_yaw | elevator | post_stall | 0.8 | 1 | 0.1834492625 | 0.6749124759 | 0.2121687841 | 11.9499891 | 2.729980863 | 10.04162593 | `
- `C7_c6_plus_alpha_regime_aileron_yaw | elevator | post_stall | 1 | 1 | 0.378964287 | 0.264030609 | 0.1086958907 | 1.027421036 | 9.104836485 | 5.331275873 | `
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
- `C8_c7_plus_alpha_regime_aileron_side_force | elevator | transition | 0.2 | 1 | 0.03514568241 | 0.1922533016 | 0.05134740803 | 23.19565304 | 5.510792551 | 12.41237255 | `
- `C8_c7_plus_alpha_regime_aileron_side_force | elevator | transition | 0.4 | 1 | 0.3345009861 | 0.006702783071 | 0.684198053 | 7.44388449 | 25.95531786 | 3.992255983 | `
- `C8_c7_plus_alpha_regime_aileron_side_force | elevator | transition | 0.6 | 1 | 0.01759335793 | 0.5835096014 | 0.03446190055 | 13.62501317 | 9.318694557 | 4.572206256 | `
- `C8_c7_plus_alpha_regime_aileron_side_force | elevator | transition | 0.8 | 1 | 0.2033673874 | 0.5724988678 | 0.01603495607 | 13.5827088 | 14.80083128 | 2.772055922 | `
- `C8_c7_plus_alpha_regime_aileron_side_force | elevator | transition | 1 | 1 | 0.1546895227 | 0.3430166752 | 0.1804882375 | 1.603723552 | 4.698571933 | 6.575000553 | `
- `C8_c7_plus_alpha_regime_aileron_side_force | elevator | post_stall | 0.2 | 1 | 0.0141268322 | 0.5801979004 | 0.05803946557 | 13.7390615 | 1.237664591 | 9.888670023 | `
- `C8_c7_plus_alpha_regime_aileron_side_force | elevator | post_stall | 0.4 | 1 | 0.4078806164 | 0.1832099995 | 0.1197822161 | 18.39421087 | 16.59185538 | 2.492769222 | `
- `C8_c7_plus_alpha_regime_aileron_side_force | elevator | post_stall | 0.6 | 1 | 0.1959485261 | 0.3863741154 | 0.1966535344 | 19.69356216 | 8.027764405 | 7.226163884 | `
- `C8_c7_plus_alpha_regime_aileron_side_force | elevator | post_stall | 0.8 | 1 | 0.1834492625 | 0.6749124759 | 0.2121687841 | 11.9499891 | 2.729980863 | 10.04162593 | `
- `C8_c7_plus_alpha_regime_aileron_side_force | elevator | post_stall | 1 | 1 | 0.378964287 | 0.264030609 | 0.1086958907 | 1.027421036 | 9.104836485 | 5.331275873 | `
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
- elevator: all `0.240740627`, high-confidence `0.2369480768` (delta `-0.00379255018`), weighted `0.2117075637` (delta `-0.02903306326`)
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

- `peak_q_rad_s` at |cmd| `0.2`: real antisym `-0.03035664165`, frozen replay antisym `0.4516536345`, symmetric `-1.794393169`
- `peak_q_rad_s` at |cmd| `0.4`: real antisym `0.1238087593`, frozen replay antisym `0.3883222439`, symmetric `-1.795648329`
- `peak_q_rad_s` at |cmd| `0.6`: real antisym `0.5185255419`, frozen replay antisym `0.7347447233`, symmetric `-1.924735809`
- `peak_q_rad_s` at |cmd| `0.8`: real antisym `0.7856975071`, frozen replay antisym `0.660974641`, symmetric `-2.207331075`
- `peak_q_rad_s` at |cmd| `1`: real antisym `0.852705801`, frozen replay antisym `0.9689431276`, symmetric `-2.184567145`
- `q_impulse_rad` at |cmd| `0.2`: real antisym `0.06495496952`, frozen replay antisym `0.1831250511`, symmetric `-0.6759196646`
- `q_impulse_rad` at |cmd| `0.4`: real antisym `0.1012180106`, frozen replay antisym `0.1540825456`, symmetric `-0.6678374679`
- `q_impulse_rad` at |cmd| `0.6`: real antisym `0.2910953912`, frozen replay antisym `0.2543108581`, symmetric `-0.7174272371`
- `q_impulse_rad` at |cmd| `0.8`: real antisym `0.3722199138`, frozen replay antisym `0.3066991731`, symmetric `-0.8612466801`
- `q_impulse_rad` at |cmd| `1`: real antisym `0.3734546679`, frozen replay antisym `0.3879061559`, symmetric `-0.7616556321`
- `theta_change_deg` at |cmd| `0.2`: real antisym `3.064137532`, frozen replay antisym `8.172674819`, symmetric `-14.58694289`
- `theta_change_deg` at |cmd| `0.4`: real antisym `7.467085683`, frozen replay antisym `11.71843877`, symmetric `-14.61867972`
- `theta_change_deg` at |cmd| `0.6`: real antisym `14.74324526`, frozen replay antisym `13.30818031`, symmetric `-14.43100039`
- `theta_change_deg` at |cmd| `0.8`: real antisym `21.66625753`, frozen replay antisym `17.92080884`, symmetric `-19.6709983`
- `theta_change_deg` at |cmd| `1`: real antisym `23.77823652`, frozen replay antisym `26.0009157`, symmetric `-20.43839507`

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

- `S0_frozen_neutral`: `evaluated_frozen_active_calibrated_model`, promoted `False`
- `S1_surface_effectiveness_scales`: `not_run_disabled_by_cli`, promoted `False`
- `S2_scales_plus_neutral_biases`: `not_run_disabled_by_cli`, promoted `False`
- `P1_pairwise_aileron_gain`: `diagnostic_pairwise_response_gain_replay_not_promoted`, promoted `False`
- `P2_pairwise_elevator_gain`: `diagnostic_pairwise_response_gain_replay_not_promoted`, promoted `False`
- `P3_pairwise_rudder_gain`: `diagnostic_pairwise_response_gain_replay_not_promoted`, promoted `False`
- `P4_pairwise_all_surface_gains`: `diagnostic_pairwise_response_gain_replay_not_promoted`, promoted `False`
- `D0_launch_confidence_weighted_derivative_fit_basis`: `diagnostic_derivative_level_fit_not_promoted`, promoted `False`
- `M0_active_elevator_aero_effectiveness_scale`: `promoted_conservative_elevator_effectiveness_only`, promoted `True`
- S1/S2 surface-scale diagnostics are disabled by default because measured surface magnitudes are used.

- Pairwise response-gain diagnostic fits `real_antisym ~= gain * frozen_sim_antisym` from confidence-weighted train ladder pairs; replay candidates P1-P4 are diagnostic only.
- `aileron` peak_p_rad_s: gain `0.846965832` (raw `0.80870729`), held-out metric baseline `0.1848068848`, candidate `0.2099364243`, improved `False`
- `elevator` peak_q_rad_s: gain `0.7704623618` (raw `0.7130779523`), held-out metric baseline `0.5415352502`, candidate `0.5260438822`, improved `True`
- `rudder` peak_r_rad_s: gain `0.5586162408` (raw `0.448270301`), held-out metric baseline `0.5716014072`, candidate `0.2861098721`, improved `True`

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
- `Cm_delta_e_residual`: coeff `-0.005154269186`, held-out baseline `0.07969507425`, candidate `0.07947109102`, improved `True`
- `Cm_delta_e_residual@normal`: coeff `0.07063069994`, held-out baseline `0.1146261881`, candidate `0.1113775303`, improved `True`
- `Cm_delta_e_residual@transition`: coeff `0.04092138088`, held-out baseline `0.06561830131`, candidate `0.06652222972`, improved `False`
- `Cm_delta_e_residual@post_stall`: coeff `-0.2252129254`, held-out baseline `0.07886193619`, candidate `0.05447424106`, improved `True`
- `CY_delta_r_residual`: coeff `-0.1708085654`, held-out baseline `0.2371454031`, candidate `0.2394389883`, improved `False`
- `Cn_delta_r_residual`: coeff `-0.01043556406`, held-out baseline `0.01338416377`, candidate `0.01290536735`, improved `True`
- `Cn_delta_r_residual@normal`: coeff `-0.0264172626`, held-out baseline `0.01219381698`, candidate `0.01428393127`, improved `False`
- `Cn_delta_r_residual@transition`: coeff `-0.01021342626`, held-out baseline `0.01346449988`, candidate `0.01292318496`, improved `True`
- `Cl_delta_r_residual`: coeff `-0.004294737192`, held-out baseline `0.01286848089`, candidate `0.01276270234`, improved `True`

## 14. Promotion Decision

Only the conservative elevator aerodynamic-effectiveness scale is promoted in the active model. Aileron/rudder effectiveness, lateral/cross-axis derivatives, and alpha-regime schedules remain diagnostic because their held-out replay trade-offs are mixed and launch-condition contaminated.

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
# Replay execution is hard-coded to 8 workers inside the study runner.
python 03_Control/02_Inner_Loop/run_control_surface_effectiveness_study.py
pytest 03_Control/tests/test_control_surface_effectiveness_study.py
pytest 03_Control/tests/test_neutral_aero_residual_sysid_contract.py
```
