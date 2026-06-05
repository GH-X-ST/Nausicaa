# Current Model Result Set Cleanup

Date: 2026-06-05

This audit records the cleanup of old neutral SysID and control-surface generated
results after promotion of the current calibrated replay model:

```text
neutral_dry_air_residual_calibrated_replay_n30_compact_coupled_elevator_rudder_effectiveness_tiny_cnbeta_heavy_sweep_v1
```

Per `docs/housekeeping_and_naming_rules.md`, historical generated result folders
are local-only unless explicitly approved for preservation and must not be kept
as a tracked archive under `03_Control/05_Results`.

Historical generated folders were moved to the ignored local archive:

```text
.codex_tmp/current_model_cleanup_20260605
```

Retained neutral replay/SysID evidence:

```text
03_Control/05_Results/glider_model_calibration_prep/n30_active_model_open_loop_sanity_audit_v1
03_Control/05_Results/glider_model_calibration_prep/n30_cmq_wide_diagnostic_v1
03_Control/05_Results/glider_model_calibration_prep/n30_compact_joint_sweep_from_active_v1
03_Control/05_Results/glider_model_calibration_prep/n30_tiny_cnbeta_diagnostic_v1
03_Control/05_Results/glider_model_calibration_prep/n30_tiny_cnbeta_heavy_sweep_v1
```

Retained control-surface evidence:

```text
03_Control/05_Results/control_surface_effectiveness/control_surface_effectiveness_v3_0_final_cnbeta
```

Active surface-effectiveness constants in
`03_Control/02_Inner_Loop/A_model_parameters/neutral_dry_air_calibration.py`:

```text
delta_a = 1.00
delta_e = 0.60
delta_r = 0.531
```

The retained control-surface result folder was generated before the final
checked-in rudder-scale promotion, so its report/manifest may still describe
rudder as diagnostic or unity. The evidence for the active rudder scale is in
`pairwise_surface_gain_fit.csv`, where the rudder bounded gain is approximately
`0.5308967863`; the active constants use the rounded conservative value `0.531`.

Related current-model replay figures:

```text
03_Control/A_figures/real_flight_replay_comparison
```
