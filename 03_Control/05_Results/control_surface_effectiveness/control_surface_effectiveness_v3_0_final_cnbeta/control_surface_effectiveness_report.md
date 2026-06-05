# Real-Flight Control Surface Effectiveness Study v3.0

## 1. Purpose and Claim Boundary

The neutral aero fit remains frozen. Deflection ladder throws support one conservative elevator aerodynamic-effectiveness correction in the active plant model; aileron, rudder, and cross-axis terms remain diagnostic. This is not broad aerodynamic SysID and does not claim accurate full 6-DoF lateral derivative identification.

- active calibrated model: `neutral_dry_air_residual_calibrated_replay_n30_compact_coupled_elevator_effectiveness_tiny_cnbeta_heavy_sweep_v1`
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
- replay dx MAE: `0.2398643992` m
- replay dy MAE: `0.4915025915` m
- replay altitude-loss MAE: `0.2142490156` m

## 5. Candidate Replay Error Summary

The candidate comparison now puts simple pairwise +/- response-gain replay first, then keeps the launch-confidence-weighted residual surface aero/coupling derivatives as diagnostic comparisons. C0 is the frozen active calibrated model, including the promoted conservative elevator effectiveness scale. P1-P4 are extra pairwise response-gain replays. C1-C8 are derivative/coupling diagnostics. No aileron, rudder, lateral/coupling, or regime-scheduled candidate is promoted by this report.

- successful candidate-family replays: `2088` / `2088`

- replay MAE comparison; lower is better:
`candidate | split | surface | dx | dy | altitude | phi | theta | psi | primary antisym`
- `C0_frozen_neutral | all | all | 0.2398643992 | 0.4915025915 | 0.2142490156 | 17.07591271 | 11.7733271 | 16.47179278 | 0.3079924361`
- `C0_frozen_neutral | all | aileron | 0.3803685688 | 0.5396754846 | 0.3170084741 | 25.06196153 | 9.487756775 | 25.99287089 | 0.1669960806`
- `C0_frozen_neutral | all | elevator | 0.148255413 | 0.436638593 | 0.1742193569 | 12.62161596 | 19.294813 | 7.662457165 | 0.2411931984`
- `C0_frozen_neutral | all | rudder | 0.1869672648 | 0.4964171986 | 0.1490573344 | 13.32967191 | 6.706498711 | 15.44400867 | 0.5157880292`
- `C0_frozen_neutral | heldout | all | 0.2498045061 | 0.4239031987 | 0.1894774146 | 16.24723839 | 13.41036105 | 18.82724416 | 0.4457862024`
- `C0_frozen_neutral | heldout | aileron | 0.4057547558 | 0.3756563777 | 0.2976731877 | 24.17594503 | 11.12158357 | 34.453587 | 0.1693178319`
- `C0_frozen_neutral | heldout | elevator | 0.1118348024 | 0.3998128477 | 0.1104923048 | 12.17130458 | 20.00420684 | 5.45521074 | 0.5431311701`
- `C0_frozen_neutral | heldout | rudder | 0.2318239601 | 0.4962403706 | 0.1602667512 | 12.39446556 | 9.105292748 | 16.57293475 | 0.6249096051`
- `P1_pairwise_aileron_gain | all | all | 0.2364728727 | 0.4917647502 | 0.2074164507 | 15.64080336 | 11.34070886 | 15.57375923 | 0.3151359892`
- `P1_pairwise_aileron_gain | all | aileron | 0.3703664398 | 0.5404486307 | 0.296858198 | 20.82960514 | 8.211899574 | 23.34443295 | 0.1884267399`
- `P1_pairwise_aileron_gain | all | elevator | 0.148255413 | 0.436638593 | 0.1742193569 | 12.62161596 | 19.294813 | 7.662457165 | 0.2411931984`
- `P1_pairwise_aileron_gain | all | rudder | 0.1869672648 | 0.4964171986 | 0.1490573344 | 13.32967191 | 6.706498711 | 15.44400867 | 0.5157880292`
- `P1_pairwise_aileron_gain | heldout | all | 0.2431017885 | 0.4241228535 | 0.1834501121 | 14.93528044 | 13.21346294 | 17.90062526 | 0.4587696394`
- `P1_pairwise_aileron_gain | heldout | aileron | 0.385646603 | 0.3763153423 | 0.2795912802 | 20.24007118 | 10.53088924 | 31.6737303 | 0.2082681431`
- `P1_pairwise_aileron_gain | heldout | elevator | 0.1118348024 | 0.3998128477 | 0.1104923048 | 12.17130458 | 20.00420684 | 5.45521074 | 0.5431311701`
- `P1_pairwise_aileron_gain | heldout | rudder | 0.2318239601 | 0.4962403706 | 0.1602667512 | 12.39446556 | 9.105292748 | 16.57293475 | 0.6249096051`
- `P2_pairwise_elevator_gain | all | all | 0.2424721951 | 0.4924717287 | 0.2155704283 | 17.23491557 | 10.82802408 | 16.40453798 | 0.3062037478`
- `P2_pairwise_elevator_gain | all | aileron | 0.3803685688 | 0.5396754846 | 0.3170084741 | 25.06196153 | 9.487756775 | 25.99287089 | 0.1669960806`
- `P2_pairwise_elevator_gain | all | elevator | 0.1562160532 | 0.4395970119 | 0.1782531433 | 13.10699312 | 16.40915115 | 7.457153041 | 0.2358271337`
- `P2_pairwise_elevator_gain | all | rudder | 0.1869672648 | 0.4964171986 | 0.1490573344 | 13.32967191 | 6.706498711 | 15.44400867 | 0.5157880292`
- `P2_pairwise_elevator_gain | heldout | all | 0.2580557582 | 0.4255896272 | 0.2066291192 | 16.57466305 | 12.50602123 | 18.72594286 | 0.5220047411`
- `P2_pairwise_elevator_gain | heldout | aileron | 0.4057547558 | 0.3756563777 | 0.2976731877 | 24.17594503 | 11.12158357 | 34.453587 | 0.1693178319`
- `P2_pairwise_elevator_gain | heldout | elevator | 0.1365885587 | 0.4048721333 | 0.1619474187 | 13.15357857 | 17.29118738 | 5.151306838 | 0.7717867863`
- `P2_pairwise_elevator_gain | heldout | rudder | 0.2318239601 | 0.4962403706 | 0.1602667512 | 12.39446556 | 9.105292748 | 16.57293475 | 0.6249096051`
- `P3_pairwise_rudder_gain | all | all | 0.2559029615 | 0.4824378161 | 0.2167228344 | 17.75189295 | 11.8525887 | 13.7104312 | 0.1824671598`
- `P3_pairwise_rudder_gain | all | aileron | 0.3803685688 | 0.5396754846 | 0.3170084741 | 25.06196153 | 9.487756775 | 25.99287089 | 0.1669960806`
- `P3_pairwise_rudder_gain | all | elevator | 0.148255413 | 0.436638593 | 0.1742193569 | 12.62161596 | 19.294813 | 7.662457165 | 0.2411931984`
- `P3_pairwise_rudder_gain | all | rudder | 0.2350829517 | 0.4692228727 | 0.1564787908 | 15.35761266 | 6.944283513 | 7.159923935 | 0.1392122002`
- `P3_pairwise_rudder_gain | heldout | all | 0.2631542352 | 0.4208089561 | 0.1878304549 | 17.24887756 | 13.1392756 | 15.89603279 | 0.3273073801`
- `P3_pairwise_rudder_gain | heldout | aileron | 0.4057547558 | 0.3756563777 | 0.2976731877 | 24.17594503 | 11.12158357 | 34.453587 | 0.1693178319`
- `P3_pairwise_rudder_gain | heldout | elevator | 0.1118348024 | 0.3998128477 | 0.1104923048 | 12.17130458 | 20.00420684 | 5.45521074 | 0.5431311701`
- `P3_pairwise_rudder_gain | heldout | rudder | 0.2718731473 | 0.486957643 | 0.1553258723 | 15.39938305 | 8.292036395 | 7.779300634 | 0.2694731384`
- `P4_pairwise_all_surface_gains | all | all | 0.2551192309 | 0.4836691121 | 0.2112116823 | 16.47578647 | 10.47466744 | 12.74514285 | 0.1878220246`
- `P4_pairwise_all_surface_gains | all | aileron | 0.3703664398 | 0.5404486307 | 0.296858198 | 20.82960514 | 8.211899574 | 23.34443295 | 0.1884267399`
- `P4_pairwise_all_surface_gains | all | elevator | 0.1562160532 | 0.4395970119 | 0.1782531433 | 13.10699312 | 16.40915115 | 7.457153041 | 0.2358271337`
- `P4_pairwise_all_surface_gains | all | rudder | 0.2350829517 | 0.4692228727 | 0.1564787908 | 15.35761266 | 6.944283513 | 7.159923935 | 0.1392122002`
- `P4_pairwise_all_surface_gains | heldout | all | 0.2647027697 | 0.4227150395 | 0.1989548571 | 16.26434427 | 12.03803767 | 14.86811259 | 0.4165093559`
- `P4_pairwise_all_surface_gains | heldout | aileron | 0.385646603 | 0.3763153423 | 0.2795912802 | 20.24007118 | 10.53088924 | 31.6737303 | 0.2082681431`
- `P4_pairwise_all_surface_gains | heldout | elevator | 0.1365885587 | 0.4048721333 | 0.1619474187 | 13.15357857 | 17.29118738 | 5.151306838 | 0.7717867863`
- `P4_pairwise_all_surface_gains | heldout | rudder | 0.2718731473 | 0.486957643 | 0.1553258723 | 15.39938305 | 8.292036395 | 7.779300634 | 0.2694731384`
- `C1_primary_moment_derivatives | all | all | 0.2528885437 | 0.4708637946 | 0.2173966598 | 16.99613014 | 11.75158393 | 12.70278016 | 0.2330212703`
- `C1_primary_moment_derivatives | all | aileron | 0.3764018823 | 0.534118681 | 0.3244945781 | 22.90205937 | 9.165670757 | 22.13198683 | 0.2926519585`
- `C1_primary_moment_derivatives | all | elevator | 0.1480390169 | 0.4368346325 | 0.1746551824 | 12.64672981 | 19.20732431 | 7.640232896 | 0.2360548394`
- `C1_primary_moment_derivatives | all | rudder | 0.2302874411 | 0.4399607592 | 0.150456781 | 15.26278522 | 7.054888681 | 8.086262939 | 0.1703570129`
- `C1_primary_moment_derivatives | heldout | all | 0.2607032684 | 0.3985736908 | 0.1861366682 | 16.44154073 | 13.66543698 | 13.99162673 | 0.4007090149`
- `C1_primary_moment_derivatives | heldout | aileron | 0.3917420918 | 0.3765745758 | 0.3001985709 | 22.67175292 | 12.14178208 | 29.98652674 | 0.2755874163`
- `C1_primary_moment_derivatives | heldout | elevator | 0.1097987353 | 0.4000177978 | 0.1123866704 | 12.18232309 | 19.97317455 | 5.430679796 | 0.5547338147`
- `C1_primary_moment_derivatives | heldout | rudder | 0.2805689782 | 0.4191286988 | 0.1458247634 | 14.47054618 | 8.881354299 | 6.557673658 | 0.3718058137`
- `C2_c1_plus_aileron_adverse_yaw | all | all | 0.2270853746 | 0.5290464628 | 0.2366688486 | 15.74031522 | 13.58812751 | 9.786306629 | 0.1917371458`
- `C2_c1_plus_aileron_adverse_yaw | all | aileron | 0.3003044005 | 0.7057082447 | 0.3813312029 | 19.19846961 | 14.58191794 | 13.53086149 | 0.1687995852`
- `C2_c1_plus_aileron_adverse_yaw | all | elevator | 0.1480390169 | 0.4368346325 | 0.1746551824 | 12.64672981 | 19.20732431 | 7.640232896 | 0.2360548394`
- `C2_c1_plus_aileron_adverse_yaw | all | rudder | 0.2302874411 | 0.4399607592 | 0.150456781 | 15.26278522 | 7.054888681 | 8.086262939 | 0.1703570129`
- `C2_c1_plus_aileron_adverse_yaw | heldout | all | 0.232458702 | 0.5203071672 | 0.1996679132 | 14.79749158 | 14.68976003 | 9.976202462 | 0.3739459923`
- `C2_c1_plus_aileron_adverse_yaw | heldout | aileron | 0.3070083926 | 0.7417750048 | 0.3407923058 | 17.73960547 | 15.21475124 | 17.94025393 | 0.1952983484`
- `C2_c1_plus_aileron_adverse_yaw | heldout | elevator | 0.1097987353 | 0.4000177978 | 0.1123866704 | 12.18232309 | 19.97317455 | 5.430679796 | 0.5547338147`
- `C2_c1_plus_aileron_adverse_yaw | heldout | rudder | 0.2805689782 | 0.4191286988 | 0.1458247634 | 14.47054618 | 8.881354299 | 6.557673658 | 0.3718058137`
- `C3_c1_plus_rudder_roll | all | all | 0.252913243 | 0.470635959 | 0.2172020759 | 17.09366405 | 11.77075614 | 12.65838116 | 0.2363977064`
- `C3_c1_plus_rudder_roll | all | aileron | 0.3764018823 | 0.534118681 | 0.3244945781 | 22.90205937 | 9.165670757 | 22.13198683 | 0.2926519585`
- `C3_c1_plus_rudder_roll | all | elevator | 0.1480390169 | 0.4368346325 | 0.1746551824 | 12.64672981 | 19.20732431 | 7.640232896 | 0.2360548394`
- `C3_c1_plus_rudder_roll | all | rudder | 0.230361539 | 0.4392772524 | 0.1498730294 | 15.55538696 | 7.112405315 | 7.953065926 | 0.1804863212`
- `C3_c1_plus_rudder_roll | heldout | all | 0.2607197223 | 0.3975959416 | 0.1858815651 | 16.45273064 | 13.67330217 | 14.05967939 | 0.4015683551`
- `C3_c1_plus_rudder_roll | heldout | aileron | 0.3917420918 | 0.3765745758 | 0.3001985709 | 22.67175292 | 12.14178208 | 29.98652674 | 0.2755874163`
- `C3_c1_plus_rudder_roll | heldout | elevator | 0.1097987353 | 0.4000177978 | 0.1123866704 | 12.18232309 | 19.97317455 | 5.430679796 | 0.5547338147`
- `C3_c1_plus_rudder_roll | heldout | rudder | 0.2806183399 | 0.4161954511 | 0.1450594541 | 14.50411589 | 8.904949887 | 6.76183164 | 0.3743838344`
- `C4_c1_plus_surface_side_force | all | all | 0.2549130215 | 0.4641062453 | 0.2075037926 | 17.05618308 | 11.50886714 | 13.35271107 | 0.2238615391`
- `C4_c1_plus_surface_side_force | all | aileron | 0.380215852 | 0.5411875246 | 0.2961513953 | 23.54671404 | 8.350793469 | 24.54156237 | 0.2825851663`
- `C4_c1_plus_surface_side_force | all | elevator | 0.1480390169 | 0.4368346325 | 0.1746551824 | 12.64672981 | 19.20732431 | 7.640232896 | 0.2360548394`
- `C4_c1_plus_surface_side_force | all | rudder | 0.2324811469 | 0.4124973912 | 0.1496100378 | 14.78717463 | 7.155665209 | 7.584935725 | 0.1529446115`
- `C4_c1_plus_surface_side_force | heldout | all | 0.2649942834 | 0.3718846889 | 0.1780655126 | 16.59258418 | 13.48546088 | 14.83854925 | 0.3894857987`
- `C4_c1_plus_surface_side_force | heldout | aileron | 0.4024862255 | 0.3225668833 | 0.2807944697 | 23.21833507 | 11.53985941 | 32.82498704 | 0.2602974695`
- `C4_c1_plus_surface_side_force | heldout | elevator | 0.1097987353 | 0.4000177978 | 0.1123866704 | 12.18232309 | 19.97317455 | 5.430679796 | 0.5547338147`
- `C4_c1_plus_surface_side_force | heldout | rudder | 0.2826978893 | 0.3930693856 | 0.1410153978 | 14.37709437 | 8.943348685 | 6.259980916 | 0.3534261118`
- `C5_c2_plus_surface_side_force | all | all | 0.2369793393 | 0.4938251243 | 0.2262079933 | 16.03650694 | 13.33531723 | 9.142099025 | 0.1871450468`
- `C5_c2_plus_surface_side_force | all | aileron | 0.3273266875 | 0.6288330321 | 0.3513129364 | 20.53953356 | 13.7372734 | 12.12382514 | 0.1724356896`
- `C5_c2_plus_surface_side_force | all | elevator | 0.1480390169 | 0.4368346325 | 0.1746551824 | 12.64672981 | 19.20732431 | 7.640232896 | 0.2360548394`
- `C5_c2_plus_surface_side_force | all | rudder | 0.2324811469 | 0.4124973912 | 0.1496100378 | 14.78717463 | 7.155665209 | 7.584935725 | 0.1529446115`
- `C5_c2_plus_surface_side_force | heldout | all | 0.240862987 | 0.4652895738 | 0.1916225557 | 15.15319959 | 14.50160809 | 9.620217673 | 0.3667358156`
- `C5_c2_plus_surface_side_force | heldout | aileron | 0.3300923365 | 0.6027815379 | 0.321465599 | 18.90018132 | 14.58830104 | 17.16999231 | 0.1920475203`
- `C5_c2_plus_surface_side_force | heldout | elevator | 0.1097987353 | 0.4000177978 | 0.1123866704 | 12.18232309 | 19.97317455 | 5.430679796 | 0.5547338147`
- `C5_c2_plus_surface_side_force | heldout | rudder | 0.2826978893 | 0.3930693856 | 0.1410153978 | 14.37709437 | 8.943348685 | 6.259980916 | 0.3534261118`
- `C6_alpha_regime_primary_derivatives | all | all | 0.2692876843 | 0.459336475 | 0.224788998 | 18.31036142 | 8.60884556 | 11.45296197 | 0.2727228313`
- `C6_alpha_regime_primary_derivatives | all | aileron | 0.3741031628 | 0.5305723245 | 0.3269637313 | 26.05088034 | 9.10159495 | 19.38258619 | 0.3782699748`
- `C6_alpha_regime_primary_derivatives | all | elevator | 0.1801766641 | 0.4405750547 | 0.1919082131 | 12.35560258 | 9.67588013 | 7.232395886 | 0.2156544381`
- `C6_alpha_regime_primary_derivatives | all | rudder | 0.2502396658 | 0.4053103688 | 0.1531665062 | 16.28847585 | 7.058963069 | 7.534417803 | 0.2242440812`
- `C6_alpha_regime_primary_derivatives | heldout | all | 0.2896467367 | 0.3787802205 | 0.2020717629 | 18.4829229 | 10.12094314 | 12.58960332 | 0.5197348066`
- `C6_alpha_regime_primary_derivatives | heldout | aileron | 0.3820364317 | 0.3781865643 | 0.3028132456 | 27.96393344 | 11.92342232 | 24.81732207 | 0.4036724652`
- `C6_alpha_regime_primary_derivatives | heldout | elevator | 0.1882444972 | 0.4064215129 | 0.1648339345 | 11.51445805 | 10.0446222 | 5.476518912 | 0.5698104714`
- `C6_alpha_regime_primary_derivatives | heldout | rudder | 0.2986592813 | 0.3517325844 | 0.1385681086 | 15.9703772 | 8.394784886 | 7.474968989 | 0.5857214831`
- `C7_c6_plus_alpha_regime_aileron_yaw | all | all | 0.2656163844 | 0.4873455355 | 0.2353404856 | 15.83853111 | 9.587178386 | 8.115427588 | 0.2182761547`
- `C7_c6_plus_alpha_regime_aileron_yaw | all | aileron | 0.3632759392 | 0.6131753163 | 0.3580816778 | 18.76107572 | 11.98684769 | 9.539688175 | 0.2149299448`
- `C7_c6_plus_alpha_regime_aileron_yaw | all | elevator | 0.1801766641 | 0.4405750547 | 0.1919082131 | 12.35560258 | 9.67588013 | 7.232395886 | 0.2156544381`
- `C7_c6_plus_alpha_regime_aileron_yaw | all | rudder | 0.2502396658 | 0.4053103688 | 0.1531665062 | 16.28847585 | 7.058963069 | 7.534417803 | 0.2242440812`
- `C7_c6_plus_alpha_regime_aileron_yaw | heldout | all | 0.2841077014 | 0.4752076013 | 0.2073966544 | 15.14483751 | 10.62101357 | 7.065767337 | 0.4494784782`
- `C7_c6_plus_alpha_regime_aileron_yaw | heldout | aileron | 0.3654193258 | 0.6674687067 | 0.3187879202 | 17.94967729 | 13.42363362 | 8.245814109 | 0.1929034802`
- `C7_c6_plus_alpha_regime_aileron_yaw | heldout | elevator | 0.1882444972 | 0.4064215129 | 0.1648339345 | 11.51445805 | 10.0446222 | 5.476518912 | 0.5698104714`
- `C7_c6_plus_alpha_regime_aileron_yaw | heldout | rudder | 0.2986592813 | 0.3517325844 | 0.1385681086 | 15.9703772 | 8.394784886 | 7.474968989 | 0.5857214831`
- `C8_c7_plus_alpha_regime_aileron_side_force | all | all | 0.2517451719 | 0.5320062299 | 0.2551994651 | 15.01428636 | 9.984546606 | 9.051604039 | 0.2101867912`
- `C8_c7_plus_alpha_regime_aileron_side_force | all | aileron | 0.3223676177 | 0.7448865169 | 0.4166488376 | 16.3302522 | 13.15874719 | 12.30061533 | 0.1906618542`
- `C8_c7_plus_alpha_regime_aileron_side_force | all | elevator | 0.1801766641 | 0.4405750547 | 0.1919082131 | 12.35560258 | 9.67588013 | 7.232395886 | 0.2156544381`
- `C8_c7_plus_alpha_regime_aileron_side_force | all | rudder | 0.2502396658 | 0.4053103688 | 0.1531665062 | 16.28847585 | 7.058963069 | 7.534417803 | 0.2242440812`
- `C8_c7_plus_alpha_regime_aileron_side_force | heldout | all | 0.270120444 | 0.4622140861 | 0.2352581274 | 14.63667201 | 10.89047071 | 8.607188043 | 0.447858358`
- `C8_c7_plus_alpha_regime_aileron_side_force | heldout | aileron | 0.3234575536 | 0.6284881611 | 0.4023723392 | 16.42518079 | 14.23200504 | 12.87007623 | 0.1880431195`
- `C8_c7_plus_alpha_regime_aileron_side_force | heldout | elevator | 0.1882444972 | 0.4064215129 | 0.1648339345 | 11.51445805 | 10.0446222 | 5.476518912 | 0.5698104714`
- `C8_c7_plus_alpha_regime_aileron_side_force | heldout | rudder | 0.2986592813 | 0.3517325844 | 0.1385681086 | 15.9703772 | 8.394784886 | 7.474968989 | 0.5857214831`

## 6. Alpha-Regime Command-Ladder Replay Error

Replay error is also reported as an explicit candidate/surface/alpha-regime/20 percent command ladder. Regime is assigned from measured response-window `actual_max_abs_alpha_deg`: normal `<12 deg`, transition `12-22 deg`, and post-stall `>=22 deg`. Empty cells are retained in `control_surface_regime_ladder_error_summary.csv` with `replay_count=0` so missing support is visible.

- held-out non-empty cells; lower is better:
`candidate | surface | regime | |cmd| | n | dx | dy | altitude | phi | theta | psi | primary`
- `C0_frozen_neutral | aileron | transition | 0.2 | 2 | 0.446713767 | 0.189465405 | 0.2391887422 | 19.13495116 | 6.232766845 | 6.099116978 | 0.1443847472`
- `C0_frozen_neutral | aileron | transition | 0.4 | 2 | 0.3759108724 | 0.5773270839 | 0.4691970311 | 16.65795618 | 11.59981298 | 9.535572295 | 0.1759104044`
- `C0_frozen_neutral | aileron | transition | 0.6 | 1 | 0.8621993464 | 0.02963607432 | 0.939475407 | 38.92074708 | 15.75545139 | 8.725150704 | `
- `C0_frozen_neutral | aileron | transition | 0.8 | 2 | 0.1413617525 | 0.2014491678 | 0.03314942796 | 20.33538873 | 7.221044163 | 46.47168818 | 0.08832714061`
- `C0_frozen_neutral | aileron | post_stall | 0.6 | 1 | 0.09839045237 | 0.2951006383 | 0.1117757868 | 12.27697809 | 2.435252143 | 69.08270276 | `
- `C0_frozen_neutral | aileron | post_stall | 1 | 2 | 0.5844924879 | 0.7476718754 | 0.2212051403 | 39.1525665 | 21.4589421 | 71.25763082 | 0.4193816807`
- `C0_frozen_neutral | elevator | transition | 0.2 | 1 | 0.02501134085 | 0.2239691783 | 0.05127714334 | 19.9100199 | 4.932500072 | 11.32006316 | `
- `C0_frozen_neutral | elevator | transition | 0.4 | 1 | 0.2235375532 | 0.009115083802 | 0.02724139762 | 13.99512783 | 16.40473838 | 3.404286766 | `
- `C0_frozen_neutral | elevator | transition | 0.6 | 1 | 0.0328043775 | 0.6207294242 | 0.01788899418 | 12.65531326 | 14.53898287 | 3.828790969 | `
- `C0_frozen_neutral | elevator | transition | 0.8 | 1 | 0.1387507909 | 0.6240260151 | 0.1570573617 | 13.72847444 | 33.80541179 | 1.637123685 | `
- `C0_frozen_neutral | elevator | transition | 1 | 1 | 0.1143818525 | 0.3646257923 | 0.481094281 | 0.2874652149 | 32.01243857 | 1.979669046 | `
- `C0_frozen_neutral | elevator | post_stall | 0.2 | 1 | 0.04787548413 | 0.6617350828 | 0.009322884636 | 10.38696038 | 7.945727773 | 7.299922888 | `
- `C0_frozen_neutral | elevator | post_stall | 0.4 | 1 | 0.3814810145 | 0.1146061468 | 0.1494094976 | 16.98643625 | 27.27012791 | 1.321507095 | `
- `C0_frozen_neutral | elevator | post_stall | 0.6 | 1 | 0.0501864913 | 0.4126542982 | 0.07250534879 | 20.02345053 | 16.94169392 | 8.190720444 | `
- `C0_frozen_neutral | elevator | post_stall | 0.8 | 1 | 0.04865565112 | 0.6757920163 | 0.1385263723 | 12.37331581 | 23.35156421 | 9.270674823 | `
- `C0_frozen_neutral | elevator | post_stall | 1 | 1 | 0.05566346755 | 0.2908754387 | 0.0005997665762 | 1.366482216 | 22.8388829 | 6.299348526 | `
- `C0_frozen_neutral | rudder | transition | 0.2 | 2 | 0.3624700835 | 0.1076670697 | 0.3596419175 | 15.09235324 | 9.024764657 | 9.361672102 | 0.7660902831`
- `C0_frozen_neutral | rudder | transition | 0.4 | 2 | 0.134435845 | 0.3534677431 | 0.08216529073 | 18.05922174 | 12.18288955 | 7.25019678 | 0.6076511443`
- `C0_frozen_neutral | rudder | transition | 0.6 | 2 | 0.1426161568 | 0.391494707 | 0.02393606415 | 10.57023027 | 1.718084375 | 10.26658555 | 0.4339820811`
- `C0_frozen_neutral | rudder | transition | 0.8 | 2 | 0.190134708 | 0.7722275998 | 0.1528979333 | 7.159898744 | 16.19676022 | 29.26347146 | 0.6432662256`
- `C0_frozen_neutral | rudder | transition | 1 | 2 | 0.3294630075 | 0.8563447336 | 0.1826925503 | 11.09062381 | 6.40396493 | 26.72274784 | 0.6735582916`
- `P1_pairwise_aileron_gain | aileron | transition | 0.2 | 2 | 0.4483323293 | 0.1861898532 | 0.2340441116 | 17.88314486 | 6.349884433 | 5.703537845 | 0.1894574695`
- `P1_pairwise_aileron_gain | aileron | transition | 0.4 | 2 | 0.3763698618 | 0.5602320904 | 0.4611558438 | 13.35322527 | 10.56780449 | 7.777460805 | 0.2688855462`
- `P1_pairwise_aileron_gain | aileron | transition | 0.6 | 1 | 0.8930692227 | 0.01576690321 | 0.8955530388 | 29.72792749 | 14.26987851 | 9.4647169 | `
- `P1_pairwise_aileron_gain | aileron | transition | 0.8 | 2 | 0.1044188771 | 0.2010784397 | 0.03770821421 | 19.42491811 | 6.90588612 | 40.58985783 | 0.1462733588`
- `P1_pairwise_aileron_gain | aileron | post_stall | 0.6 | 1 | 0.05069856853 | 0.3391233846 | 0.1063067969 | 12.40809265 | 3.298488521 | 62.94289987 | `
- `P1_pairwise_aileron_gain | aileron | post_stall | 1 | 2 | 0.5272280514 | 0.7566311844 | 0.1641183136 | 29.4710576 | 20.04668762 | 68.09398665 | 0.2905209791`
- `P1_pairwise_aileron_gain | elevator | transition | 0.2 | 1 | 0.02501134085 | 0.2239691783 | 0.05127714334 | 19.9100199 | 4.932500072 | 11.32006316 | `
- `P1_pairwise_aileron_gain | elevator | transition | 0.4 | 1 | 0.2235375532 | 0.009115083802 | 0.02724139762 | 13.99512783 | 16.40473838 | 3.404286766 | `
- `P1_pairwise_aileron_gain | elevator | transition | 0.6 | 1 | 0.0328043775 | 0.6207294242 | 0.01788899418 | 12.65531326 | 14.53898287 | 3.828790969 | `
- `P1_pairwise_aileron_gain | elevator | transition | 0.8 | 1 | 0.1387507909 | 0.6240260151 | 0.1570573617 | 13.72847444 | 33.80541179 | 1.637123685 | `
- `P1_pairwise_aileron_gain | elevator | transition | 1 | 1 | 0.1143818525 | 0.3646257923 | 0.481094281 | 0.2874652149 | 32.01243857 | 1.979669046 | `
- `P1_pairwise_aileron_gain | elevator | post_stall | 0.2 | 1 | 0.04787548413 | 0.6617350828 | 0.009322884636 | 10.38696038 | 7.945727773 | 7.299922888 | `
- `P1_pairwise_aileron_gain | elevator | post_stall | 0.4 | 1 | 0.3814810145 | 0.1146061468 | 0.1494094976 | 16.98643625 | 27.27012791 | 1.321507095 | `
- `P1_pairwise_aileron_gain | elevator | post_stall | 0.6 | 1 | 0.0501864913 | 0.4126542982 | 0.07250534879 | 20.02345053 | 16.94169392 | 8.190720444 | `
- `P1_pairwise_aileron_gain | elevator | post_stall | 0.8 | 1 | 0.04865565112 | 0.6757920163 | 0.1385263723 | 12.37331581 | 23.35156421 | 9.270674823 | `
- `P1_pairwise_aileron_gain | elevator | post_stall | 1 | 1 | 0.05566346755 | 0.2908754387 | 0.0005997665762 | 1.366482216 | 22.8388829 | 6.299348526 | `
- `P1_pairwise_aileron_gain | rudder | transition | 0.2 | 2 | 0.3624700835 | 0.1076670697 | 0.3596419175 | 15.09235324 | 9.024764657 | 9.361672102 | 0.7660902831`
- `P1_pairwise_aileron_gain | rudder | transition | 0.4 | 2 | 0.134435845 | 0.3534677431 | 0.08216529073 | 18.05922174 | 12.18288955 | 7.25019678 | 0.6076511443`
- `P1_pairwise_aileron_gain | rudder | transition | 0.6 | 2 | 0.1426161568 | 0.391494707 | 0.02393606415 | 10.57023027 | 1.718084375 | 10.26658555 | 0.4339820811`
- `P1_pairwise_aileron_gain | rudder | transition | 0.8 | 2 | 0.190134708 | 0.7722275998 | 0.1528979333 | 7.159898744 | 16.19676022 | 29.26347146 | 0.6432662256`
- `P1_pairwise_aileron_gain | rudder | transition | 1 | 2 | 0.3294630075 | 0.8563447336 | 0.1826925503 | 11.09062381 | 6.40396493 | 26.72274784 | 0.6735582916`
- `P2_pairwise_elevator_gain | aileron | transition | 0.2 | 2 | 0.446713767 | 0.189465405 | 0.2391887422 | 19.13495116 | 6.232766845 | 6.099116978 | 0.1443847472`
- `P2_pairwise_elevator_gain | aileron | transition | 0.4 | 2 | 0.3759108724 | 0.5773270839 | 0.4691970311 | 16.65795618 | 11.59981298 | 9.535572295 | 0.1759104044`
- `P2_pairwise_elevator_gain | aileron | transition | 0.6 | 1 | 0.8621993464 | 0.02963607432 | 0.939475407 | 38.92074708 | 15.75545139 | 8.725150704 | `
- `P2_pairwise_elevator_gain | aileron | transition | 0.8 | 2 | 0.1413617525 | 0.2014491678 | 0.03314942796 | 20.33538873 | 7.221044163 | 46.47168818 | 0.08832714061`
- `P2_pairwise_elevator_gain | aileron | post_stall | 0.6 | 1 | 0.09839045237 | 0.2951006383 | 0.1117757868 | 12.27697809 | 2.435252143 | 69.08270276 | `
- `P2_pairwise_elevator_gain | aileron | post_stall | 1 | 2 | 0.5844924879 | 0.7476718754 | 0.2212051403 | 39.1525665 | 21.4589421 | 71.25763082 | 0.4193816807`
- `P2_pairwise_elevator_gain | elevator | transition | 0.2 | 1 | 0.01280172608 | 0.2224956135 | 0.04653927395 | 18.30054771 | 4.156913082 | 11.36500179 | `
- `P2_pairwise_elevator_gain | elevator | transition | 0.4 | 1 | 0.3578886802 | 0.03953080328 | 0.4388328484 | 6.755722888 | 14.30964316 | 3.834241325 | `
- `P2_pairwise_elevator_gain | elevator | transition | 0.6 | 1 | 0.03860696267 | 0.6211482623 | 0.01915881843 | 14.01027975 | 16.02065025 | 3.705150806 | `
- `P2_pairwise_elevator_gain | elevator | transition | 0.8 | 1 | 0.1229673663 | 0.6204651993 | 0.1689755052 | 20.50080992 | 36.5924914 | 1.264870176 | `
- `P2_pairwise_elevator_gain | elevator | transition | 1 | 1 | 0.07159328948 | 0.3725037367 | 0.5271157698 | 11.540617 | 38.2617318 | 1.984164771 | `
- `P2_pairwise_elevator_gain | elevator | post_stall | 0.2 | 1 | 0.03452057883 | 0.6634198279 | 0.02509962181 | 10.39176992 | 5.440660486 | 7.145803291 | `
- `P2_pairwise_elevator_gain | elevator | post_stall | 0.4 | 1 | 0.4118507198 | 0.1108709398 | 0.1285162315 | 16.50694535 | 22.42937904 | 0.2686459956 | `
- `P2_pairwise_elevator_gain | elevator | post_stall | 0.6 | 1 | 0.06622328368 | 0.4200491958 | 0.1039703478 | 20.6186708 | 8.775323773 | 7.121264318 | `
- `P2_pairwise_elevator_gain | elevator | post_stall | 0.8 | 1 | 0.07942787953 | 0.6839637616 | 0.1555152697 | 12.23828569 | 13.89668237 | 9.566984814 | `
- `P2_pairwise_elevator_gain | elevator | post_stall | 1 | 1 | 0.1700051008 | 0.2942739927 | 0.005750500613 | 0.6721366511 | 13.0283985 | 5.256941089 | `
- `P2_pairwise_elevator_gain | rudder | transition | 0.2 | 2 | 0.3624700835 | 0.1076670697 | 0.3596419175 | 15.09235324 | 9.024764657 | 9.361672102 | 0.7660902831`
- `P2_pairwise_elevator_gain | rudder | transition | 0.4 | 2 | 0.134435845 | 0.3534677431 | 0.08216529073 | 18.05922174 | 12.18288955 | 7.25019678 | 0.6076511443`
- `P2_pairwise_elevator_gain | rudder | transition | 0.6 | 2 | 0.1426161568 | 0.391494707 | 0.02393606415 | 10.57023027 | 1.718084375 | 10.26658555 | 0.4339820811`
- `P2_pairwise_elevator_gain | rudder | transition | 0.8 | 2 | 0.190134708 | 0.7722275998 | 0.1528979333 | 7.159898744 | 16.19676022 | 29.26347146 | 0.6432662256`
- `P2_pairwise_elevator_gain | rudder | transition | 1 | 2 | 0.3294630075 | 0.8563447336 | 0.1826925503 | 11.09062381 | 6.40396493 | 26.72274784 | 0.6735582916`
- `P3_pairwise_rudder_gain | aileron | transition | 0.2 | 2 | 0.446713767 | 0.189465405 | 0.2391887422 | 19.13495116 | 6.232766845 | 6.099116978 | 0.1443847472`
- `P3_pairwise_rudder_gain | aileron | transition | 0.4 | 2 | 0.3759108724 | 0.5773270839 | 0.4691970311 | 16.65795618 | 11.59981298 | 9.535572295 | 0.1759104044`
- `P3_pairwise_rudder_gain | aileron | transition | 0.6 | 1 | 0.8621993464 | 0.02963607432 | 0.939475407 | 38.92074708 | 15.75545139 | 8.725150704 | `
- `P3_pairwise_rudder_gain | aileron | transition | 0.8 | 2 | 0.1413617525 | 0.2014491678 | 0.03314942796 | 20.33538873 | 7.221044163 | 46.47168818 | 0.08832714061`
- `P3_pairwise_rudder_gain | aileron | post_stall | 0.6 | 1 | 0.09839045237 | 0.2951006383 | 0.1117757868 | 12.27697809 | 2.435252143 | 69.08270276 | `
- `P3_pairwise_rudder_gain | aileron | post_stall | 1 | 2 | 0.5844924879 | 0.7476718754 | 0.2212051403 | 39.1525665 | 21.4589421 | 71.25763082 | 0.4193816807`
- `P3_pairwise_rudder_gain | elevator | transition | 0.2 | 1 | 0.02501134085 | 0.2239691783 | 0.05127714334 | 19.9100199 | 4.932500072 | 11.32006316 | `
- `P3_pairwise_rudder_gain | elevator | transition | 0.4 | 1 | 0.2235375532 | 0.009115083802 | 0.02724139762 | 13.99512783 | 16.40473838 | 3.404286766 | `
- `P3_pairwise_rudder_gain | elevator | transition | 0.6 | 1 | 0.0328043775 | 0.6207294242 | 0.01788899418 | 12.65531326 | 14.53898287 | 3.828790969 | `
- `P3_pairwise_rudder_gain | elevator | transition | 0.8 | 1 | 0.1387507909 | 0.6240260151 | 0.1570573617 | 13.72847444 | 33.80541179 | 1.637123685 | `
- `P3_pairwise_rudder_gain | elevator | transition | 1 | 1 | 0.1143818525 | 0.3646257923 | 0.481094281 | 0.2874652149 | 32.01243857 | 1.979669046 | `
- `P3_pairwise_rudder_gain | elevator | post_stall | 0.2 | 1 | 0.04787548413 | 0.6617350828 | 0.009322884636 | 10.38696038 | 7.945727773 | 7.299922888 | `
- `P3_pairwise_rudder_gain | elevator | post_stall | 0.4 | 1 | 0.3814810145 | 0.1146061468 | 0.1494094976 | 16.98643625 | 27.27012791 | 1.321507095 | `
- `P3_pairwise_rudder_gain | elevator | post_stall | 0.6 | 1 | 0.0501864913 | 0.4126542982 | 0.07250534879 | 20.02345053 | 16.94169392 | 8.190720444 | `
- `P3_pairwise_rudder_gain | elevator | post_stall | 0.8 | 1 | 0.04865565112 | 0.6757920163 | 0.1385263723 | 12.37331581 | 23.35156421 | 9.270674823 | `
- `P3_pairwise_rudder_gain | elevator | post_stall | 1 | 1 | 0.05566346755 | 0.2908754387 | 0.0005997665762 | 1.366482216 | 22.8388829 | 6.299348526 | `
- `P3_pairwise_rudder_gain | rudder | transition | 0.2 | 2 | 0.3720725044 | 0.09693300702 | 0.3551688935 | 15.60347745 | 8.514188489 | 5.2060883 | 0.8535240791`
- `P3_pairwise_rudder_gain | rudder | transition | 0.4 | 2 | 0.143595775 | 0.3991107028 | 0.08212203929 | 20.17270438 | 12.39953326 | 6.373565269 | 0.3545849535`
- `P3_pairwise_rudder_gain | rudder | transition | 0.6 | 2 | 0.1218455505 | 0.4464942483 | 0.02111357797 | 12.25096171 | 2.552602222 | 1.351124252 | 0.0647891874`
- `P3_pairwise_rudder_gain | rudder | transition | 0.8 | 2 | 0.2699455873 | 0.6920510501 | 0.1542324788 | 12.15727055 | 13.92137216 | 15.41903504 | 0.03547986279`
- `P3_pairwise_rudder_gain | rudder | transition | 1 | 2 | 0.4519063196 | 0.8001992067 | 0.1639923719 | 16.81250118 | 4.072485839 | 10.54669031 | 0.03898760908`
- `P4_pairwise_all_surface_gains | aileron | transition | 0.2 | 2 | 0.4483323293 | 0.1861898532 | 0.2340441116 | 17.88314486 | 6.349884433 | 5.703537845 | 0.1894574695`
- `P4_pairwise_all_surface_gains | aileron | transition | 0.4 | 2 | 0.3763698618 | 0.5602320904 | 0.4611558438 | 13.35322527 | 10.56780449 | 7.777460805 | 0.2688855462`
- `P4_pairwise_all_surface_gains | aileron | transition | 0.6 | 1 | 0.8930692227 | 0.01576690321 | 0.8955530388 | 29.72792749 | 14.26987851 | 9.4647169 | `
- `P4_pairwise_all_surface_gains | aileron | transition | 0.8 | 2 | 0.1044188771 | 0.2010784397 | 0.03770821421 | 19.42491811 | 6.90588612 | 40.58985783 | 0.1462733588`
- `P4_pairwise_all_surface_gains | aileron | post_stall | 0.6 | 1 | 0.05069856853 | 0.3391233846 | 0.1063067969 | 12.40809265 | 3.298488521 | 62.94289987 | `
- `P4_pairwise_all_surface_gains | aileron | post_stall | 1 | 2 | 0.5272280514 | 0.7566311844 | 0.1641183136 | 29.4710576 | 20.04668762 | 68.09398665 | 0.2905209791`
- `P4_pairwise_all_surface_gains | elevator | transition | 0.2 | 1 | 0.01280172608 | 0.2224956135 | 0.04653927395 | 18.30054771 | 4.156913082 | 11.36500179 | `
- `P4_pairwise_all_surface_gains | elevator | transition | 0.4 | 1 | 0.3578886802 | 0.03953080328 | 0.4388328484 | 6.755722888 | 14.30964316 | 3.834241325 | `
- `P4_pairwise_all_surface_gains | elevator | transition | 0.6 | 1 | 0.03860696267 | 0.6211482623 | 0.01915881843 | 14.01027975 | 16.02065025 | 3.705150806 | `
- `P4_pairwise_all_surface_gains | elevator | transition | 0.8 | 1 | 0.1229673663 | 0.6204651993 | 0.1689755052 | 20.50080992 | 36.5924914 | 1.264870176 | `
- `P4_pairwise_all_surface_gains | elevator | transition | 1 | 1 | 0.07159328948 | 0.3725037367 | 0.5271157698 | 11.540617 | 38.2617318 | 1.984164771 | `
- `P4_pairwise_all_surface_gains | elevator | post_stall | 0.2 | 1 | 0.03452057883 | 0.6634198279 | 0.02509962181 | 10.39176992 | 5.440660486 | 7.145803291 | `
- `P4_pairwise_all_surface_gains | elevator | post_stall | 0.4 | 1 | 0.4118507198 | 0.1108709398 | 0.1285162315 | 16.50694535 | 22.42937904 | 0.2686459956 | `
- `P4_pairwise_all_surface_gains | elevator | post_stall | 0.6 | 1 | 0.06622328368 | 0.4200491958 | 0.1039703478 | 20.6186708 | 8.775323773 | 7.121264318 | `
- `P4_pairwise_all_surface_gains | elevator | post_stall | 0.8 | 1 | 0.07942787953 | 0.6839637616 | 0.1555152697 | 12.23828569 | 13.89668237 | 9.566984814 | `
- `P4_pairwise_all_surface_gains | elevator | post_stall | 1 | 1 | 0.1700051008 | 0.2942739927 | 0.005750500613 | 0.6721366511 | 13.0283985 | 5.256941089 | `
- `P4_pairwise_all_surface_gains | rudder | transition | 0.2 | 2 | 0.3720725044 | 0.09693300702 | 0.3551688935 | 15.60347745 | 8.514188489 | 5.2060883 | 0.8535240791`
- `P4_pairwise_all_surface_gains | rudder | transition | 0.4 | 2 | 0.143595775 | 0.3991107028 | 0.08212203929 | 20.17270438 | 12.39953326 | 6.373565269 | 0.3545849535`
- `P4_pairwise_all_surface_gains | rudder | transition | 0.6 | 2 | 0.1218455505 | 0.4464942483 | 0.02111357797 | 12.25096171 | 2.552602222 | 1.351124252 | 0.0647891874`
- `P4_pairwise_all_surface_gains | rudder | transition | 0.8 | 2 | 0.2699455873 | 0.6920510501 | 0.1542324788 | 12.15727055 | 13.92137216 | 15.41903504 | 0.03547986279`
- `P4_pairwise_all_surface_gains | rudder | transition | 1 | 2 | 0.4519063196 | 0.8001992067 | 0.1639923719 | 16.81250118 | 4.072485839 | 10.54669031 | 0.03898760908`
- `C1_primary_moment_derivatives | aileron | transition | 0.2 | 2 | 0.4456520275 | 0.1746072729 | 0.2310775943 | 19.47374476 | 6.72718772 | 4.804362175 | 0.2749155963`
- `C1_primary_moment_derivatives | aileron | transition | 0.4 | 2 | 0.3692670799 | 0.5454476412 | 0.4621322376 | 12.73191279 | 10.64065552 | 7.493258679 | 0.3372549741`
- `C1_primary_moment_derivatives | aileron | transition | 0.6 | 1 | 0.8584480396 | 0.0253030817 | 0.9187059521 | 28.12676874 | 14.1965565 | 11.15495576 | `
- `C1_primary_moment_derivatives | aileron | transition | 0.8 | 2 | 0.1236966407 | 0.1875997617 | 0.02538871034 | 22.36342754 | 4.910458964 | 38.19901185 | 0.2983651804`
- `C1_primary_moment_derivatives | aileron | post_stall | 0.6 | 1 | 0.06258553764 | 0.3476546036 | 0.113785779 | 21.86593223 | 6.129137177 | 60.02399535 | `
- `C1_primary_moment_derivatives | aileron | post_stall | 1 | 2 | 0.5595779225 | 0.7887393605 | 0.2661484466 | 33.79332903 | 28.26776135 | 63.84652544 | 0.1824184168`
- `C1_primary_moment_derivatives | elevator | transition | 0.2 | 1 | 0.02383377095 | 0.2239703602 | 0.05104458645 | 19.73717759 | 4.870621857 | 11.31762106 | `
- `C1_primary_moment_derivatives | elevator | transition | 0.4 | 1 | 0.2183106937 | 0.009613291653 | 0.02859019608 | 13.96720201 | 16.11174364 | 3.537123372 | `
- `C1_primary_moment_derivatives | elevator | transition | 0.6 | 1 | 0.03390956521 | 0.6206957928 | 0.01695674088 | 12.74675226 | 14.77592448 | 3.828243719 | `
- `C1_primary_moment_derivatives | elevator | transition | 0.8 | 1 | 0.1352815435 | 0.6234745994 | 0.1624526472 | 13.86502975 | 34.42906086 | 1.655482569 | `
- `C1_primary_moment_derivatives | elevator | transition | 1 | 1 | 0.1072586121 | 0.3659690323 | 0.4941099933 | 0.4037029058 | 33.39435571 | 1.81213943 | `
- `C1_primary_moment_derivatives | elevator | post_stall | 0.2 | 1 | 0.04678705612 | 0.6618562251 | 0.01046903077 | 10.38598896 | 7.765130376 | 7.289335515 | `
- `C1_primary_moment_derivatives | elevator | post_stall | 0.4 | 1 | 0.386697298 | 0.1141045202 | 0.1472235427 | 16.9683648 | 26.90926587 | 1.22841237 | `
- `C1_primary_moment_derivatives | elevator | post_stall | 0.6 | 1 | 0.0423293374 | 0.4130827227 | 0.07381680149 | 20.06360214 | 16.43558019 | 8.120384869 | `
- `C1_primary_moment_derivatives | elevator | post_stall | 0.8 | 1 | 0.0403300701 | 0.6763368555 | 0.1388557699 | 12.36555468 | 22.78946915 | 9.288083469 | `
- `C1_primary_moment_derivatives | elevator | post_stall | 1 | 1 | 0.06324940574 | 0.2910745783 | 0.0003473948013 | 1.319855858 | 22.25059338 | 6.229971586 | `
- `C1_primary_moment_derivatives | rudder | transition | 0.2 | 2 | 0.3668609169 | 0.08382566389 | 0.356194042 | 15.39765682 | 8.769909731 | 7.325723067 | 0.8089431775`
- `C1_primary_moment_derivatives | rudder | transition | 0.4 | 2 | 0.1372618019 | 0.382681987 | 0.08098041202 | 19.21093437 | 12.28379958 | 5.389104493 | 0.4744833195`
- `C1_primary_moment_derivatives | rudder | transition | 0.6 | 2 | 0.1383647307 | 0.4323250792 | 0.02323422429 | 11.67132767 | 2.354144056 | 5.115804236 | 0.232217911`
- `C1_primary_moment_derivatives | rudder | transition | 0.8 | 2 | 0.268131523 | 0.615735451 | 0.1422420395 | 10.8737028 | 15.27159013 | 13.9539973 | 0.1397678005`
- `C1_primary_moment_derivatives | rudder | transition | 1 | 2 | 0.4922259186 | 0.5810753131 | 0.1264730991 | 15.19910925 | 5.727328 | 1.003739194 | 0.20361686`
- `C2_c1_plus_aileron_adverse_yaw | aileron | transition | 0.2 | 2 | 0.4283827043 | 0.3159160657 | 0.2551232669 | 18.18222881 | 6.767595053 | 9.070883602 | 0.1890197296`
- `C2_c1_plus_aileron_adverse_yaw | aileron | transition | 0.4 | 2 | 0.3360220836 | 0.7292058503 | 0.4976607416 | 17.5587685 | 13.14185782 | 12.96592029 | 0.1649367038`
- `C2_c1_plus_aileron_adverse_yaw | aileron | transition | 0.6 | 1 | 0.7142539013 | 0.9241563889 | 1.036633842 | 30.29306803 | 27.55367568 | 25.89709677 | `
- `C2_c1_plus_aileron_adverse_yaw | aileron | transition | 0.8 | 2 | 0.07229803763 | 0.3468740713 | 0.02239472946 | 21.43792154 | 3.756941541 | 5.254325109 | 0.1910550672`
- `C2_c1_plus_aileron_adverse_yaw | aileron | post_stall | 0.6 | 1 | 0.01850232716 | 0.6856723566 | 0.09733302497 | 0.3017123059 | 7.584908771 | 26.10294418 | `
- `C2_c1_plus_aileron_adverse_yaw | aileron | post_stall | 1 | 2 | 0.3319610231 | 1.511964664 | 0.3617993577 | 16.22171835 | 34.83806955 | 36.41012018 | 0.4020497119`
- `C2_c1_plus_aileron_adverse_yaw | elevator | transition | 0.2 | 1 | 0.02383377095 | 0.2239703602 | 0.05104458645 | 19.73717759 | 4.870621857 | 11.31762106 | `
- `C2_c1_plus_aileron_adverse_yaw | elevator | transition | 0.4 | 1 | 0.2183106937 | 0.009613291653 | 0.02859019608 | 13.96720201 | 16.11174364 | 3.537123372 | `
- `C2_c1_plus_aileron_adverse_yaw | elevator | transition | 0.6 | 1 | 0.03390956521 | 0.6206957928 | 0.01695674088 | 12.74675226 | 14.77592448 | 3.828243719 | `
- `C2_c1_plus_aileron_adverse_yaw | elevator | transition | 0.8 | 1 | 0.1352815435 | 0.6234745994 | 0.1624526472 | 13.86502975 | 34.42906086 | 1.655482569 | `
- `C2_c1_plus_aileron_adverse_yaw | elevator | transition | 1 | 1 | 0.1072586121 | 0.3659690323 | 0.4941099933 | 0.4037029058 | 33.39435571 | 1.81213943 | `
- `C2_c1_plus_aileron_adverse_yaw | elevator | post_stall | 0.2 | 1 | 0.04678705612 | 0.6618562251 | 0.01046903077 | 10.38598896 | 7.765130376 | 7.289335515 | `
- `C2_c1_plus_aileron_adverse_yaw | elevator | post_stall | 0.4 | 1 | 0.386697298 | 0.1141045202 | 0.1472235427 | 16.9683648 | 26.90926587 | 1.22841237 | `
- `C2_c1_plus_aileron_adverse_yaw | elevator | post_stall | 0.6 | 1 | 0.0423293374 | 0.4130827227 | 0.07381680149 | 20.06360214 | 16.43558019 | 8.120384869 | `
- `C2_c1_plus_aileron_adverse_yaw | elevator | post_stall | 0.8 | 1 | 0.0403300701 | 0.6763368555 | 0.1388557699 | 12.36555468 | 22.78946915 | 9.288083469 | `
- `C2_c1_plus_aileron_adverse_yaw | elevator | post_stall | 1 | 1 | 0.06324940574 | 0.2910745783 | 0.0003473948013 | 1.319855858 | 22.25059338 | 6.229971586 | `
- `C2_c1_plus_aileron_adverse_yaw | rudder | transition | 0.2 | 2 | 0.3668609169 | 0.08382566389 | 0.356194042 | 15.39765682 | 8.769909731 | 7.325723067 | 0.8089431775`
- `C2_c1_plus_aileron_adverse_yaw | rudder | transition | 0.4 | 2 | 0.1372618019 | 0.382681987 | 0.08098041202 | 19.21093437 | 12.28379958 | 5.389104493 | 0.4744833195`
- `C2_c1_plus_aileron_adverse_yaw | rudder | transition | 0.6 | 2 | 0.1383647307 | 0.4323250792 | 0.02323422429 | 11.67132767 | 2.354144056 | 5.115804236 | 0.232217911`
- `C2_c1_plus_aileron_adverse_yaw | rudder | transition | 0.8 | 2 | 0.268131523 | 0.615735451 | 0.1422420395 | 10.8737028 | 15.27159013 | 13.9539973 | 0.1397678005`
- `C2_c1_plus_aileron_adverse_yaw | rudder | transition | 1 | 2 | 0.4922259186 | 0.5810753131 | 0.1264730991 | 15.19910925 | 5.727328 | 1.003739194 | 0.20361686`
- `C3_c1_plus_rudder_roll | aileron | transition | 0.2 | 2 | 0.4456520275 | 0.1746072729 | 0.2310775943 | 19.47374476 | 6.72718772 | 4.804362175 | 0.2749155963`
- `C3_c1_plus_rudder_roll | aileron | transition | 0.4 | 2 | 0.3692670799 | 0.5454476412 | 0.4621322376 | 12.73191279 | 10.64065552 | 7.493258679 | 0.3372549741`
- `C3_c1_plus_rudder_roll | aileron | transition | 0.6 | 1 | 0.8584480396 | 0.0253030817 | 0.9187059521 | 28.12676874 | 14.1965565 | 11.15495576 | `
- `C3_c1_plus_rudder_roll | aileron | transition | 0.8 | 2 | 0.1236966407 | 0.1875997617 | 0.02538871034 | 22.36342754 | 4.910458964 | 38.19901185 | 0.2983651804`
- `C3_c1_plus_rudder_roll | aileron | post_stall | 0.6 | 1 | 0.06258553764 | 0.3476546036 | 0.113785779 | 21.86593223 | 6.129137177 | 60.02399535 | `
- `C3_c1_plus_rudder_roll | aileron | post_stall | 1 | 2 | 0.5595779225 | 0.7887393605 | 0.2661484466 | 33.79332903 | 28.26776135 | 63.84652544 | 0.1824184168`
- `C3_c1_plus_rudder_roll | elevator | transition | 0.2 | 1 | 0.02383377095 | 0.2239703602 | 0.05104458645 | 19.73717759 | 4.870621857 | 11.31762106 | `
- `C3_c1_plus_rudder_roll | elevator | transition | 0.4 | 1 | 0.2183106937 | 0.009613291653 | 0.02859019608 | 13.96720201 | 16.11174364 | 3.537123372 | `
- `C3_c1_plus_rudder_roll | elevator | transition | 0.6 | 1 | 0.03390956521 | 0.6206957928 | 0.01695674088 | 12.74675226 | 14.77592448 | 3.828243719 | `
- `C3_c1_plus_rudder_roll | elevator | transition | 0.8 | 1 | 0.1352815435 | 0.6234745994 | 0.1624526472 | 13.86502975 | 34.42906086 | 1.655482569 | `
- `C3_c1_plus_rudder_roll | elevator | transition | 1 | 1 | 0.1072586121 | 0.3659690323 | 0.4941099933 | 0.4037029058 | 33.39435571 | 1.81213943 | `
- `C3_c1_plus_rudder_roll | elevator | post_stall | 0.2 | 1 | 0.04678705612 | 0.6618562251 | 0.01046903077 | 10.38598896 | 7.765130376 | 7.289335515 | `
- `C3_c1_plus_rudder_roll | elevator | post_stall | 0.4 | 1 | 0.386697298 | 0.1141045202 | 0.1472235427 | 16.9683648 | 26.90926587 | 1.22841237 | `
- `C3_c1_plus_rudder_roll | elevator | post_stall | 0.6 | 1 | 0.0423293374 | 0.4130827227 | 0.07381680149 | 20.06360214 | 16.43558019 | 8.120384869 | `
- `C3_c1_plus_rudder_roll | elevator | post_stall | 0.8 | 1 | 0.0403300701 | 0.6763368555 | 0.1388557699 | 12.36555468 | 22.78946915 | 9.288083469 | `
- `C3_c1_plus_rudder_roll | elevator | post_stall | 1 | 1 | 0.06324940574 | 0.2910745783 | 0.0003473948013 | 1.319855858 | 22.25059338 | 6.229971586 | `
- `C3_c1_plus_rudder_roll | rudder | transition | 0.2 | 2 | 0.3669081824 | 0.08321858807 | 0.3557999936 | 15.41220736 | 8.757424122 | 7.346194694 | 0.8094114445`
- `C3_c1_plus_rudder_roll | rudder | transition | 0.4 | 2 | 0.137285739 | 0.3835526062 | 0.08076665191 | 19.50373218 | 12.35824164 | 5.430042132 | 0.4784843244`
- `C3_c1_plus_rudder_roll | rudder | transition | 0.6 | 2 | 0.1390022054 | 0.4298254207 | 0.02355817286 | 11.88862082 | 2.447072209 | 5.645432621 | 0.2409269681`
- `C3_c1_plus_rudder_roll | rudder | transition | 0.8 | 2 | 0.2685345387 | 0.6126418877 | 0.1405324684 | 11.30917441 | 15.10887873 | 13.90546298 | 0.1361650955`
- `C3_c1_plus_rudder_roll | rudder | transition | 1 | 2 | 0.491361034 | 0.5717387531 | 0.1246399836 | 14.4068447 | 5.853132736 | 1.482025771 | 0.2069313396`
- `C4_c1_plus_surface_side_force | aileron | transition | 0.2 | 2 | 0.4450330869 | 0.1423547729 | 0.2241048361 | 19.83687795 | 6.950296681 | 4.064912514 | 0.2678000012`
- `C4_c1_plus_surface_side_force | aileron | transition | 0.4 | 2 | 0.3770876536 | 0.5062691188 | 0.4405133583 | 13.51281699 | 10.25328761 | 10.55049985 | 0.320206649`
- `C4_c1_plus_surface_side_force | aileron | transition | 0.6 | 1 | 0.8420684607 | 0.310551223 | 0.8515427717 | 30.67209197 | 12.10644646 | 17.21417482 | `
- `C4_c1_plus_surface_side_force | aileron | transition | 0.8 | 2 | 0.137936852 | 0.1905177 | 0.01754278588 | 22.48587063 | 4.984090821 | 40.58394673 | 0.3056143431`
- `C4_c1_plus_surface_side_force | aileron | post_stall | 0.6 | 1 | 0.09215627666 | 0.2078239159 | 0.1368027944 | 21.31560068 | 5.925591765 | 61.97806822 | `
- `C4_c1_plus_surface_side_force | aileron | post_stall | 1 | 2 | 0.5852611665 | 0.5145052553 | 0.227638585 | 34.26226346 | 26.49560284 | 69.3294546 | 0.1531916829`
- `C4_c1_plus_surface_side_force | elevator | transition | 0.2 | 1 | 0.02383377095 | 0.2239703602 | 0.05104458645 | 19.73717759 | 4.870621857 | 11.31762106 | `
- `C4_c1_plus_surface_side_force | elevator | transition | 0.4 | 1 | 0.2183106937 | 0.009613291653 | 0.02859019608 | 13.96720201 | 16.11174364 | 3.537123372 | `
- `C4_c1_plus_surface_side_force | elevator | transition | 0.6 | 1 | 0.03390956521 | 0.6206957928 | 0.01695674088 | 12.74675226 | 14.77592448 | 3.828243719 | `
- `C4_c1_plus_surface_side_force | elevator | transition | 0.8 | 1 | 0.1352815435 | 0.6234745994 | 0.1624526472 | 13.86502975 | 34.42906086 | 1.655482569 | `
- `C4_c1_plus_surface_side_force | elevator | transition | 1 | 1 | 0.1072586121 | 0.3659690323 | 0.4941099933 | 0.4037029058 | 33.39435571 | 1.81213943 | `
- `C4_c1_plus_surface_side_force | elevator | post_stall | 0.2 | 1 | 0.04678705612 | 0.6618562251 | 0.01046903077 | 10.38598896 | 7.765130376 | 7.289335515 | `
- `C4_c1_plus_surface_side_force | elevator | post_stall | 0.4 | 1 | 0.386697298 | 0.1141045202 | 0.1472235427 | 16.9683648 | 26.90926587 | 1.22841237 | `
- `C4_c1_plus_surface_side_force | elevator | post_stall | 0.6 | 1 | 0.0423293374 | 0.4130827227 | 0.07381680149 | 20.06360214 | 16.43558019 | 8.120384869 | `
- `C4_c1_plus_surface_side_force | elevator | post_stall | 0.8 | 1 | 0.0403300701 | 0.6763368555 | 0.1388557699 | 12.36555468 | 22.78946915 | 9.288083469 | `
- `C4_c1_plus_surface_side_force | elevator | post_stall | 1 | 1 | 0.06324940574 | 0.2910745783 | 0.0003473948013 | 1.319855858 | 22.25059338 | 6.229971586 | `
- `C4_c1_plus_surface_side_force | rudder | transition | 0.2 | 2 | 0.3685975954 | 0.0896579633 | 0.3530499441 | 15.58047646 | 8.723814952 | 6.768896788 | 0.8167778975`
- `C4_c1_plus_surface_side_force | rudder | transition | 0.4 | 2 | 0.1383916996 | 0.4169756789 | 0.07843400276 | 19.35332576 | 12.19293168 | 5.691929133 | 0.4438075597`
- `C4_c1_plus_surface_side_force | rudder | transition | 0.6 | 2 | 0.137679111 | 0.4816894002 | 0.02609342916 | 11.62317815 | 2.172656485 | 4.052100903 | 0.1943760479`
- `C4_c1_plus_surface_side_force | rudder | transition | 0.8 | 2 | 0.2726527098 | 0.5387708696 | 0.1294087868 | 10.69907338 | 15.54302413 | 14.05221259 | 0.1225104516`
- `C4_c1_plus_surface_side_force | rudder | transition | 1 | 2 | 0.4961683305 | 0.4382530163 | 0.1180908262 | 14.62941812 | 6.084316172 | 0.7347651723 | 0.1896586025`
- `C5_c2_plus_surface_side_force | aileron | transition | 0.2 | 2 | 0.4335933485 | 0.2838200383 | 0.2475562591 | 18.58135293 | 6.525720048 | 8.303355284 | 0.1819382673`
- `C5_c2_plus_surface_side_force | aileron | transition | 0.4 | 2 | 0.3542716037 | 0.6921066367 | 0.4749606303 | 18.5625492 | 12.64387163 | 9.603996341 | 0.1511216307`
- `C5_c2_plus_surface_side_force | aileron | transition | 0.6 | 1 | 0.7887701659 | 0.6491072939 | 0.9722190139 | 33.61721581 | 25.99010675 | 18.89890607 | `
- `C5_c2_plus_surface_side_force | aileron | transition | 0.8 | 2 | 0.07093449933 | 0.204180398 | 0.00497392251 | 21.87776275 | 3.903131838 | 9.005281038 | 0.1763606211`
- `C5_c2_plus_surface_side_force | aileron | post_stall | 0.6 | 1 | 0.004780875447 | 0.5379113908 | 0.1288684773 | 0.1933988968 | 7.418559041 | 30.26067951 | `
- `C5_c2_plus_surface_side_force | aileron | post_stall | 1 | 2 | 0.3948867104 | 1.240291274 | 0.3292934376 | 18.57393435 | 33.16444878 | 34.35753608 | 0.4057467365`
- `C5_c2_plus_surface_side_force | elevator | transition | 0.2 | 1 | 0.02383377095 | 0.2239703602 | 0.05104458645 | 19.73717759 | 4.870621857 | 11.31762106 | `
- `C5_c2_plus_surface_side_force | elevator | transition | 0.4 | 1 | 0.2183106937 | 0.009613291653 | 0.02859019608 | 13.96720201 | 16.11174364 | 3.537123372 | `
- `C5_c2_plus_surface_side_force | elevator | transition | 0.6 | 1 | 0.03390956521 | 0.6206957928 | 0.01695674088 | 12.74675226 | 14.77592448 | 3.828243719 | `
- `C5_c2_plus_surface_side_force | elevator | transition | 0.8 | 1 | 0.1352815435 | 0.6234745994 | 0.1624526472 | 13.86502975 | 34.42906086 | 1.655482569 | `
- `C5_c2_plus_surface_side_force | elevator | transition | 1 | 1 | 0.1072586121 | 0.3659690323 | 0.4941099933 | 0.4037029058 | 33.39435571 | 1.81213943 | `
- `C5_c2_plus_surface_side_force | elevator | post_stall | 0.2 | 1 | 0.04678705612 | 0.6618562251 | 0.01046903077 | 10.38598896 | 7.765130376 | 7.289335515 | `
- `C5_c2_plus_surface_side_force | elevator | post_stall | 0.4 | 1 | 0.386697298 | 0.1141045202 | 0.1472235427 | 16.9683648 | 26.90926587 | 1.22841237 | `
- `C5_c2_plus_surface_side_force | elevator | post_stall | 0.6 | 1 | 0.0423293374 | 0.4130827227 | 0.07381680149 | 20.06360214 | 16.43558019 | 8.120384869 | `
- `C5_c2_plus_surface_side_force | elevator | post_stall | 0.8 | 1 | 0.0403300701 | 0.6763368555 | 0.1388557699 | 12.36555468 | 22.78946915 | 9.288083469 | `
- `C5_c2_plus_surface_side_force | elevator | post_stall | 1 | 1 | 0.06324940574 | 0.2910745783 | 0.0003473948013 | 1.319855858 | 22.25059338 | 6.229971586 | `
- `C5_c2_plus_surface_side_force | rudder | transition | 0.2 | 2 | 0.3685975954 | 0.0896579633 | 0.3530499441 | 15.58047646 | 8.723814952 | 6.768896788 | 0.8167778975`
- `C5_c2_plus_surface_side_force | rudder | transition | 0.4 | 2 | 0.1383916996 | 0.4169756789 | 0.07843400276 | 19.35332576 | 12.19293168 | 5.691929133 | 0.4438075597`
- `C5_c2_plus_surface_side_force | rudder | transition | 0.6 | 2 | 0.137679111 | 0.4816894002 | 0.02609342916 | 11.62317815 | 2.172656485 | 4.052100903 | 0.1943760479`
- `C5_c2_plus_surface_side_force | rudder | transition | 0.8 | 2 | 0.2726527098 | 0.5387708696 | 0.1294087868 | 10.69907338 | 15.54302413 | 14.05221259 | 0.1225104516`
- `C5_c2_plus_surface_side_force | rudder | transition | 1 | 2 | 0.4961683305 | 0.4382530163 | 0.1180908262 | 14.62941812 | 6.084316172 | 0.7347651723 | 0.1896586025`
- `C6_alpha_regime_primary_derivatives | aileron | transition | 0.2 | 2 | 0.4459962918 | 0.1741257329 | 0.2314286349 | 19.09636606 | 6.681510631 | 4.901273659 | 0.2661788098`
- `C6_alpha_regime_primary_derivatives | aileron | transition | 0.4 | 2 | 0.3692789373 | 0.5320018495 | 0.4617814685 | 13.96626819 | 10.13778313 | 6.782100193 | 0.3619494903`
- `C6_alpha_regime_primary_derivatives | aileron | transition | 0.6 | 1 | 0.8582807074 | 0.02864192231 | 0.9182916 | 27.52842363 | 14.13870393 | 11.30373614 | `
- `C6_alpha_regime_primary_derivatives | aileron | transition | 0.8 | 2 | 0.1120344233 | 0.1570735319 | 0.02636023809 | 34.72970063 | 2.581815486 | 29.25040201 | 0.7922877854`
- `C6_alpha_regime_primary_derivatives | aileron | post_stall | 0.6 | 1 | 0.04870619405 | 0.3738885653 | 0.1164504286 | 31.09084075 | 8.221022199 | 50.26461923 | `
- `C6_alpha_regime_primary_derivatives | aileron | post_stall | 1 | 2 | 0.5293790552 | 0.8264664634 | 0.2771248723 | 42.71770014 | 29.03613932 | 52.36865678 | 0.1758704224`
- `C6_alpha_regime_primary_derivatives | elevator | transition | 0.2 | 1 | 0.03408563499 | 0.2236095306 | 0.05215535288 | 21.36677872 | 5.454090209 | 11.37404091 | `
- `C6_alpha_regime_primary_derivatives | elevator | transition | 0.4 | 1 | 0.3302549174 | 0.0527541686 | 0.6338365189 | 7.00382147 | 24.70324929 | 4.772213056 | `
- `C6_alpha_regime_primary_derivatives | elevator | transition | 0.6 | 1 | 0.02402320054 | 0.6213653117 | 0.03570883052 | 12.08389055 | 9.311395957 | 3.686073929 | `
- `C6_alpha_regime_primary_derivatives | elevator | transition | 0.8 | 1 | 0.1821270425 | 0.6307030414 | 0.04375084811 | 13.0655775 | 18.571718 | 1.329509507 | `
- `C6_alpha_regime_primary_derivatives | elevator | transition | 1 | 1 | 0.1550125485 | 0.3307702757 | 0.1843557414 | 1.429972009 | 4.451319897 | 7.04552109 | `
- `C6_alpha_regime_primary_derivatives | elevator | post_stall | 0.2 | 1 | 0.002689270555 | 0.6671125793 | 0.05646557571 | 10.69595888 | 1.12837272 | 6.821004068 | `
- `C6_alpha_regime_primary_derivatives | elevator | post_stall | 0.4 | 1 | 0.3979649611 | 0.1110616006 | 0.1219591174 | 16.27278247 | 16.73679595 | 0.7479691089 | `
- `C6_alpha_regime_primary_derivatives | elevator | post_stall | 0.6 | 1 | 0.1946391989 | 0.4326507832 | 0.1967928486 | 21.02630822 | 8.047572609 | 5.342949789 | `
- `C6_alpha_regime_primary_derivatives | elevator | post_stall | 0.8 | 1 | 0.1822521702 | 0.6890198229 | 0.2130588797 | 12.02500452 | 2.735830568 | 10.10338736 | `
- `C6_alpha_regime_primary_derivatives | elevator | post_stall | 1 | 1 | 0.3793960274 | 0.3051680153 | 0.1102556319 | 0.1744861051 | 9.305876813 | 3.542520309 | `
- `C6_alpha_regime_primary_derivatives | rudder | transition | 0.2 | 2 | 0.3692977426 | 0.09543825738 | 0.3534672081 | 15.40075815 | 8.606860938 | 6.173395434 | 0.8351086061`
- `C6_alpha_regime_primary_derivatives | rudder | transition | 0.4 | 2 | 0.1379852355 | 0.4009722493 | 0.07982341717 | 20.29508565 | 12.32506231 | 6.387575616 | 0.4630417312`
- `C6_alpha_regime_primary_derivatives | rudder | transition | 0.6 | 2 | 0.1384109706 | 0.4317074363 | 0.02323992717 | 11.65392561 | 2.345946615 | 5.194835796 | 0.2352921769`
- `C6_alpha_regime_primary_derivatives | rudder | transition | 0.8 | 2 | 0.3087805795 | 0.4987668017 | 0.125340439 | 14.41577932 | 11.42687014 | 2.217779385 | 0.582963874`
- `C6_alpha_regime_primary_derivatives | rudder | transition | 1 | 2 | 0.5388218783 | 0.3317781772 | 0.1109695515 | 18.08633727 | 7.26918443 | 17.40125871 | 0.8122010275`
- `C7_c6_plus_alpha_regime_aileron_yaw | aileron | transition | 0.2 | 2 | 0.4428601853 | 0.2315708991 | 0.2422927647 | 17.68746205 | 6.283044958 | 5.495535021 | 0.2204372341`
- `C7_c6_plus_alpha_regime_aileron_yaw | aileron | transition | 0.4 | 2 | 0.3689096831 | 0.5140159979 | 0.4812876327 | 18.49875189 | 10.76339413 | 8.965270177 | 0.1886054101`
- `C7_c6_plus_alpha_regime_aileron_yaw | aileron | transition | 0.6 | 1 | 0.8583568683 | 0.4102573819 | 0.9746917737 | 29.18209699 | 20.09805438 | 5.35739358 | `
- `C7_c6_plus_alpha_regime_aileron_yaw | aileron | transition | 0.8 | 2 | 0.1046138578 | 0.5023299198 | 0.03919095754 | 22.99184789 | 1.839047218 | 12.30421266 | 0.2357265948`
- `C7_c6_plus_alpha_regime_aileron_yaw | aileron | post_stall | 0.6 | 1 | 0.006479311236 | 0.8238354386 | 0.07279899561 | 0.6849170825 | 11.0311579 | 12.60299019 | `
- `C7_c6_plus_alpha_regime_aileron_yaw | aileron | post_stall | 1 | 2 | 0.4782948131 | 1.472380307 | 0.3074228613 | 15.6368176 | 32.66807566 | 5.4838608 | 0.1431140284`
- `C7_c6_plus_alpha_regime_aileron_yaw | elevator | transition | 0.2 | 1 | 0.03408563499 | 0.2236095306 | 0.05215535288 | 21.36677872 | 5.454090209 | 11.37404091 | `
- `C7_c6_plus_alpha_regime_aileron_yaw | elevator | transition | 0.4 | 1 | 0.3302549174 | 0.0527541686 | 0.6338365189 | 7.00382147 | 24.70324929 | 4.772213056 | `
- `C7_c6_plus_alpha_regime_aileron_yaw | elevator | transition | 0.6 | 1 | 0.02402320054 | 0.6213653117 | 0.03570883052 | 12.08389055 | 9.311395957 | 3.686073929 | `
- `C7_c6_plus_alpha_regime_aileron_yaw | elevator | transition | 0.8 | 1 | 0.1821270425 | 0.6307030414 | 0.04375084811 | 13.0655775 | 18.571718 | 1.329509507 | `
- `C7_c6_plus_alpha_regime_aileron_yaw | elevator | transition | 1 | 1 | 0.1550125485 | 0.3307702757 | 0.1843557414 | 1.429972009 | 4.451319897 | 7.04552109 | `
- `C7_c6_plus_alpha_regime_aileron_yaw | elevator | post_stall | 0.2 | 1 | 0.002689270555 | 0.6671125793 | 0.05646557571 | 10.69595888 | 1.12837272 | 6.821004068 | `
- `C7_c6_plus_alpha_regime_aileron_yaw | elevator | post_stall | 0.4 | 1 | 0.3979649611 | 0.1110616006 | 0.1219591174 | 16.27278247 | 16.73679595 | 0.7479691089 | `
- `C7_c6_plus_alpha_regime_aileron_yaw | elevator | post_stall | 0.6 | 1 | 0.1946391989 | 0.4326507832 | 0.1967928486 | 21.02630822 | 8.047572609 | 5.342949789 | `
- `C7_c6_plus_alpha_regime_aileron_yaw | elevator | post_stall | 0.8 | 1 | 0.1822521702 | 0.6890198229 | 0.2130588797 | 12.02500452 | 2.735830568 | 10.10338736 | `
- `C7_c6_plus_alpha_regime_aileron_yaw | elevator | post_stall | 1 | 1 | 0.3793960274 | 0.3051680153 | 0.1102556319 | 0.1744861051 | 9.305876813 | 3.542520309 | `
- `C7_c6_plus_alpha_regime_aileron_yaw | rudder | transition | 0.2 | 2 | 0.3692977426 | 0.09543825738 | 0.3534672081 | 15.40075815 | 8.606860938 | 6.173395434 | 0.8351086061`
- `C7_c6_plus_alpha_regime_aileron_yaw | rudder | transition | 0.4 | 2 | 0.1379852355 | 0.4009722493 | 0.07982341717 | 20.29508565 | 12.32506231 | 6.387575616 | 0.4630417312`
- `C7_c6_plus_alpha_regime_aileron_yaw | rudder | transition | 0.6 | 2 | 0.1384109706 | 0.4317074363 | 0.02323992717 | 11.65392561 | 2.345946615 | 5.194835796 | 0.2352921769`
- `C7_c6_plus_alpha_regime_aileron_yaw | rudder | transition | 0.8 | 2 | 0.3087805795 | 0.4987668017 | 0.125340439 | 14.41577932 | 11.42687014 | 2.217779385 | 0.582963874`
- `C7_c6_plus_alpha_regime_aileron_yaw | rudder | transition | 1 | 2 | 0.5388218783 | 0.3317781772 | 0.1109695515 | 18.08633727 | 7.26918443 | 17.40125871 | 0.8122010275`
- `C8_c7_plus_alpha_regime_aileron_side_force | aileron | transition | 0.2 | 2 | 0.4352680622 | 0.3212003886 | 0.2587658528 | 16.85910742 | 6.128982515 | 7.584538748 | 0.2315066364`
- `C8_c7_plus_alpha_regime_aileron_side_force | aileron | transition | 0.4 | 2 | 0.3384925259 | 0.8532470712 | 0.5008030819 | 16.83187826 | 11.08453764 | 8.353550357 | 0.2027063111`
- `C8_c7_plus_alpha_regime_aileron_side_force | aileron | transition | 0.6 | 1 | 0.7735641798 | 0.9383190313 | 1.071417627 | 23.64734316 | 22.66953522 | 17.71549203 | `
- `C8_c7_plus_alpha_regime_aileron_side_force | aileron | transition | 0.8 | 2 | 0.07954921469 | 0.2341668934 | 0.05732646645 | 23.75812651 | 2.73609225 | 6.760403887 | 0.1489219761`
- `C8_c7_plus_alpha_regime_aileron_side_force | aileron | post_stall | 0.6 | 1 | 0.01978595515 | 0.2357309982 | 0.2117301224 | 2.409945464 | 10.81286079 | 24.29628885 | `
- `C8_c7_plus_alpha_regime_aileron_side_force | aileron | post_stall | 1 | 2 | 0.3673028976 | 1.146801438 | 0.5533924204 | 11.64814743 | 34.46921481 | 20.64599771 | 0.2329632575`
- `C8_c7_plus_alpha_regime_aileron_side_force | elevator | transition | 0.2 | 1 | 0.03408563499 | 0.2236095306 | 0.05215535288 | 21.36677872 | 5.454090209 | 11.37404091 | `
- `C8_c7_plus_alpha_regime_aileron_side_force | elevator | transition | 0.4 | 1 | 0.3302549174 | 0.0527541686 | 0.6338365189 | 7.00382147 | 24.70324929 | 4.772213056 | `
- `C8_c7_plus_alpha_regime_aileron_side_force | elevator | transition | 0.6 | 1 | 0.02402320054 | 0.6213653117 | 0.03570883052 | 12.08389055 | 9.311395957 | 3.686073929 | `
- `C8_c7_plus_alpha_regime_aileron_side_force | elevator | transition | 0.8 | 1 | 0.1821270425 | 0.6307030414 | 0.04375084811 | 13.0655775 | 18.571718 | 1.329509507 | `
- `C8_c7_plus_alpha_regime_aileron_side_force | elevator | transition | 1 | 1 | 0.1550125485 | 0.3307702757 | 0.1843557414 | 1.429972009 | 4.451319897 | 7.04552109 | `
- `C8_c7_plus_alpha_regime_aileron_side_force | elevator | post_stall | 0.2 | 1 | 0.002689270555 | 0.6671125793 | 0.05646557571 | 10.69595888 | 1.12837272 | 6.821004068 | `
- `C8_c7_plus_alpha_regime_aileron_side_force | elevator | post_stall | 0.4 | 1 | 0.3979649611 | 0.1110616006 | 0.1219591174 | 16.27278247 | 16.73679595 | 0.7479691089 | `
- `C8_c7_plus_alpha_regime_aileron_side_force | elevator | post_stall | 0.6 | 1 | 0.1946391989 | 0.4326507832 | 0.1967928486 | 21.02630822 | 8.047572609 | 5.342949789 | `
- `C8_c7_plus_alpha_regime_aileron_side_force | elevator | post_stall | 0.8 | 1 | 0.1822521702 | 0.6890198229 | 0.2130588797 | 12.02500452 | 2.735830568 | 10.10338736 | `
- `C8_c7_plus_alpha_regime_aileron_side_force | elevator | post_stall | 1 | 1 | 0.3793960274 | 0.3051680153 | 0.1102556319 | 0.1744861051 | 9.305876813 | 3.542520309 | `
- `C8_c7_plus_alpha_regime_aileron_side_force | rudder | transition | 0.2 | 2 | 0.3692977426 | 0.09543825738 | 0.3534672081 | 15.40075815 | 8.606860938 | 6.173395434 | 0.8351086061`
- `C8_c7_plus_alpha_regime_aileron_side_force | rudder | transition | 0.4 | 2 | 0.1379852355 | 0.4009722493 | 0.07982341717 | 20.29508565 | 12.32506231 | 6.387575616 | 0.4630417312`
- `C8_c7_plus_alpha_regime_aileron_side_force | rudder | transition | 0.6 | 2 | 0.1384109706 | 0.4317074363 | 0.02323992717 | 11.65392561 | 2.345946615 | 5.194835796 | 0.2352921769`
- `C8_c7_plus_alpha_regime_aileron_side_force | rudder | transition | 0.8 | 2 | 0.3087805795 | 0.4987668017 | 0.125340439 | 14.41577932 | 11.42687014 | 2.217779385 | 0.582963874`
- `C8_c7_plus_alpha_regime_aileron_side_force | rudder | transition | 1 | 2 | 0.5388218783 | 0.3317781772 | 0.1109695515 | 18.08633727 | 7.26918443 | 17.40125871 | 0.8122010275`

## 7. Launch-Confidence Diagnostic

Launch confidence is a diagnostic weight and grouping variable, not a new acceptance gate. It reuses the neutral SysID lateral-contamination strategy with reference `phi0=psi0=v0=p0=r0=0`, so the study can test whether real-vs-replay mismatch is launch-condition driven.

- all successful replays: `174` total, `139` high-confidence, `35` medium-confidence, `0` low-confidence; mean confidence weight `0.8246298746`, mean lateral-contamination score `0.3490437321`
- primary antisymmetric residual check; lower is better, negative delta means the confidence subset reduced mismatch:
- aileron: all `0.1669960806`, high-confidence `0.2490156126` (delta `0.08201953199`), weighted `0.1697852247` (delta `0.002789144043`)
- elevator: all `0.2411931984`, high-confidence `0.2376097967` (delta `-0.003583401766`), weighted `0.2119669531` (delta `-0.02922624538`)
- rudder: all `0.5157880292`, high-confidence `0.5013215549` (delta `-0.01446647431`), weighted `0.5150519803` (delta `-0.0007360489327`)

## 8. Aileron Effectiveness

- `p_impulse_rad` at |cmd| `0.2`: real antisym `-0.1210443046`, frozen replay antisym `-0.1544507743`, symmetric `-0.07303182681`
- `p_impulse_rad` at |cmd| `0.4`: real antisym `-0.2205070193`, frozen replay antisym `-0.3219142455`, symmetric `-0.1208255662`
- `p_impulse_rad` at |cmd| `0.6`: real antisym `-0.2625987419`, frozen replay antisym `-0.5055672527`, symmetric `-0.2407683994`
- `p_impulse_rad` at |cmd| `0.8`: real antisym `-0.3616444479`, frozen replay antisym `-0.448398959`, symmetric `-0.2237233768`
- `p_impulse_rad` at |cmd| `1`: real antisym `-0.4255499444`, frozen replay antisym `-0.5062196507`, symmetric `-0.110179692`
- `peak_p_rad_s` at |cmd| `0.2`: real antisym `-0.3594961665`, frozen replay antisym `-0.2479471296`, symmetric `-0.1930355039`
- `peak_p_rad_s` at |cmd| `0.4`: real antisym `-0.5748971197`, frozen replay antisym `-0.6423628396`, symmetric `-0.2362419743`
- `peak_p_rad_s` at |cmd| `0.6`: real antisym `-0.5340032405`, frozen replay antisym `-1.12356101`, symmetric `-0.5629066085`
- `peak_p_rad_s` at |cmd| `0.8`: real antisym `-0.9441721259`, frozen replay antisym `-0.9371692012`, symmetric `-0.6744709449`
- `peak_p_rad_s` at |cmd| `1`: real antisym `-1.220090835`, frozen replay antisym `-1.160685883`, symmetric `-0.2853769045`
- `phi_change_deg` at |cmd| `0.2`: real antisym `-4.978640333`, frozen replay antisym `-7.575479744`, symmetric `-7.779638421`
- `phi_change_deg` at |cmd| `0.4`: real antisym `-12.7797718`, frozen replay antisym `-16.61115771`, symmetric `-10.48163817`
- `phi_change_deg` at |cmd| `0.6`: real antisym `-17.26749192`, frozen replay antisym `-27.44988983`, symmetric `-12.57800562`
- `phi_change_deg` at |cmd| `0.8`: real antisym `-21.46354104`, frozen replay antisym `-27.31098256`, symmetric `-10.47514676`
- `phi_change_deg` at |cmd| `1`: real antisym `-23.93518186`, frozen replay antisym `-25.65156481`, symmetric `-9.27870387`

## 9. Elevator Effectiveness

- `peak_q_rad_s` at |cmd| `0.2`: real antisym `-0.03035664165`, frozen replay antisym `0.4494167568`, symmetric `-1.794393169`
- `peak_q_rad_s` at |cmd| `0.4`: real antisym `0.1238087593`, frozen replay antisym `0.3889919005`, symmetric `-1.795648329`
- `peak_q_rad_s` at |cmd| `0.6`: real antisym `0.5185255419`, frozen replay antisym `0.7369435334`, symmetric `-1.924735809`
- `peak_q_rad_s` at |cmd| `0.8`: real antisym `0.7856975071`, frozen replay antisym `0.657793955`, symmetric `-2.207331075`
- `peak_q_rad_s` at |cmd| `1`: real antisym `0.852705801`, frozen replay antisym `0.9673937099`, symmetric `-2.184567145`
- `q_impulse_rad` at |cmd| `0.2`: real antisym `0.06495496952`, frozen replay antisym `0.1825817094`, symmetric `-0.6759196646`
- `q_impulse_rad` at |cmd| `0.4`: real antisym `0.1012180106`, frozen replay antisym `0.1527999716`, symmetric `-0.6678374679`
- `q_impulse_rad` at |cmd| `0.6`: real antisym `0.2910953912`, frozen replay antisym `0.2552477044`, symmetric `-0.7174272371`
- `q_impulse_rad` at |cmd| `0.8`: real antisym `0.3722199138`, frozen replay antisym `0.3049589217`, symmetric `-0.8612466801`
- `q_impulse_rad` at |cmd| `1`: real antisym `0.3734546679`, frozen replay antisym `0.3857449557`, symmetric `-0.7616556321`
- `theta_change_deg` at |cmd| `0.2`: real antisym `3.064137532`, frozen replay antisym `8.200837903`, symmetric `-14.58694289`
- `theta_change_deg` at |cmd| `0.4`: real antisym `7.467085683`, frozen replay antisym `11.71874664`, symmetric `-14.61867972`
- `theta_change_deg` at |cmd| `0.6`: real antisym `14.74324526`, frozen replay antisym `13.33721059`, symmetric `-14.43100039`
- `theta_change_deg` at |cmd| `0.8`: real antisym `21.66625753`, frozen replay antisym `17.79780925`, symmetric `-19.6709983`
- `theta_change_deg` at |cmd| `1`: real antisym `23.77823652`, frozen replay antisym `26.00752146`, symmetric `-20.43839507`

## 10. Rudder Effectiveness

- `peak_r_rad_s` at |cmd| `0.2`: real antisym `0.0751527378`, frozen replay antisym `0.2417451623`, symmetric `0.2855270265`
- `peak_r_rad_s` at |cmd| `0.4`: real antisym `0.2858049148`, frozen replay antisym `0.6170594921`, symmetric `0.3242412438`
- `peak_r_rad_s` at |cmd| `0.6`: real antisym `0.3224312498`, frozen replay antisym `0.9770221236`, symmetric `0.3591352943`
- `peak_r_rad_s` at |cmd| `0.8`: real antisym `0.6973117494`, frozen replay antisym `1.209450638`, symmetric `0.1871974089`
- `peak_r_rad_s` at |cmd| `1`: real antisym `0.5984712171`, frozen replay antisym `1.512834599`, symmetric `0.3827552196`
- `psi_change_deg` at |cmd| `0.2`: real antisym `1.220462177`, frozen replay antisym `6.947913001`, symmetric `10.75778676`
- `psi_change_deg` at |cmd| `0.4`: real antisym `4.856261679`, frozen replay antisym `12.44013208`, symmetric `6.424287793`
- `psi_change_deg` at |cmd| `0.6`: real antisym `7.001545713`, frozen replay antisym `18.39122492`, symmetric `5.594880747`
- `psi_change_deg` at |cmd| `0.8`: real antisym `10.61294899`, frozen replay antisym `21.77972903`, symmetric `6.932046819`
- `psi_change_deg` at |cmd| `1`: real antisym `11.17866776`, frozen replay antisym `26.55532754`, symmetric `6.719284376`
- `r_impulse_rad` at |cmd| `0.2`: real antisym `0.008366203487`, frozen replay antisym `0.1001311752`, symmetric `0.1668012808`
- `r_impulse_rad` at |cmd| `0.4`: real antisym `0.0724427302`, frozen replay antisym `0.2240608394`, symmetric `0.1557162515`
- `r_impulse_rad` at |cmd| `0.6`: real antisym `0.1142859051`, frozen replay antisym `0.3470637891`, symmetric `0.1583579135`
- `r_impulse_rad` at |cmd| `0.8`: real antisym `0.1892594758`, frozen replay antisym `0.4003075559`, symmetric `0.1753381778`
- `r_impulse_rad` at |cmd| `1`: real antisym `0.1857516084`, frozen replay antisym `0.4787834319`, symmetric `0.2223284655`

## 11. Cross-Coupling Observations

Aileron yaw response and rudder roll response are reported as diagnostic coupling evidence. They are not promoted as lateral transition aerodynamic derivatives by this study.

## 12. Symmetric Launch/Trim Contamination

Symmetric response is separated from antisymmetric response. Large symmetric terms are interpreted as launch, trim, hardware, or model-mismatch contamination rather than hidden inside a surface effectiveness scale.

- aileron: mean absolute primary symmetric response `0.3904063872`
- elevator: mean absolute primary symmetric response `1.981335105`
- rudder: mean absolute primary symmetric response `0.3077712386`

## 13. Optional Surface/Aero Fit Result

- `S0_frozen_neutral`: `evaluated_frozen_active_calibrated_model`, promoted `False`
- `S1_surface_effectiveness_scales`: `diagnostic_metric_space_estimate_not_promoted`, promoted `False`
- `S2_scales_plus_neutral_biases`: `not_run_neutral_bias_not_fit_from_symmetric_contamination`, promoted `False`
- `P1_pairwise_aileron_gain`: `diagnostic_pairwise_response_gain_replay_not_promoted`, promoted `False`
- `P2_pairwise_elevator_gain`: `diagnostic_pairwise_response_gain_replay_not_promoted`, promoted `False`
- `P3_pairwise_rudder_gain`: `diagnostic_pairwise_response_gain_replay_not_promoted`, promoted `False`
- `P4_pairwise_all_surface_gains`: `diagnostic_pairwise_response_gain_replay_not_promoted`, promoted `False`
- `D0_launch_confidence_weighted_derivative_fit_basis`: `diagnostic_derivative_level_fit_not_promoted`, promoted `False`
- `M0_active_elevator_aero_effectiveness_scale`: `promoted_conservative_elevator_effectiveness_only`, promoted `True`
- S1 held-out metric diagnostics improved `41` / `75` rows, but remain not promoted.

- Pairwise response-gain diagnostic fits `real_antisym ~= gain * frozen_sim_antisym` from confidence-weighted train ladder pairs; replay candidates P1-P4 are diagnostic only.
- `aileron` peak_p_rad_s: gain `0.8621351` (raw `0.827668875`), held-out metric baseline `0.1693178319`, candidate `0.1940918426`, improved `False`
- `elevator` peak_q_rad_s: gain `0.7705171814` (raw `0.7131464768`), held-out metric baseline `0.5431311701`, candidate `0.5272772656`, improved `True`
- `rudder` peak_r_rad_s: gain `0.5308967863` (raw `0.4136209828`), held-out metric baseline `0.6249096051`, candidate `0.2934595116`, improved `True`

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
- `Cn_delta_a_residual`: coeff `-0.08537369226`, held-out baseline `0.01910032289`, candidate `0.0131325855`, improved `True`
- `Cn_delta_a_residual@normal`: coeff `-0.038422725`, held-out baseline `0.01009928603`, candidate `0.009034946387`, improved `True`
- `Cn_delta_a_residual@transition`: coeff `-0.08423782112`, held-out baseline `0.01919558842`, candidate `0.01327918105`, improved `True`
- `Cn_delta_a_residual@post_stall`: coeff `-0.1408691749`, held-out baseline `0.04816438057`, candidate `0.01133543523`, improved `True`
- `Cm_delta_e_residual`: coeff `-0.005154269186`, held-out baseline `0.07969507425`, candidate `0.07947109102`, improved `True`
- `Cm_delta_e_residual@normal`: coeff `0.07063069994`, held-out baseline `0.1146261881`, candidate `0.1113775303`, improved `True`
- `Cm_delta_e_residual@transition`: coeff `0.04092138088`, held-out baseline `0.06561830131`, candidate `0.06652222972`, improved `False`
- `Cm_delta_e_residual@post_stall`: coeff `-0.2252129254`, held-out baseline `0.07886193619`, candidate `0.05447424106`, improved `True`
- `CY_delta_r_residual`: coeff `-0.1708085654`, held-out baseline `0.2371454031`, candidate `0.2394389883`, improved `False`
- `Cn_delta_r_residual`: coeff `-0.01492262641`, held-out baseline `0.01381331412`, candidate `0.01307872405`, improved `True`
- `Cn_delta_r_residual@normal`: coeff `-0.02648548921`, held-out baseline `0.01254034054`, candidate `0.01481242753`, improved `False`
- `Cn_delta_r_residual@transition`: coeff `-0.01469439476`, held-out baseline `0.01389922669`, candidate `0.01305080733`, improved `True`
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
