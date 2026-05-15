# 03_Control Cleanup Inventory

ECT layer sequence: Cleanup -> Exploration -> Candidate

This inventory was created before deleting retired agile/OCP30/old TVLQR files. It classifies active Python files under `03_Control/` and records the cleanup action taken.

| Path | Classification | Reason | Old agile/TVLQR imports | Action taken |
|---|---|---|---|---|
| `03_Control/02_Inner_Loop/A_model_parameters/build_mass_properties_estimate.py` | DEFER_UNTOUCHED | Mass-property utility outside active control rewrite. | None found. | Leave untouched. |
| `03_Control/02_Inner_Loop/A_model_parameters/mass_properties_estimate.py` | DEFER_UNTOUCHED | Generated/parameter utility outside active control rewrite. | None found. | Leave untouched. |
| `03_Control/02_Inner_Loop/aero_model.py` | DEFER_UNTOUCHED | Legacy aero utility outside frozen active path. | None found. | Leave untouched. |
| `03_Control/02_Inner_Loop/flight_dynamics.py` | KEEP_CORE | Frozen plant derivative, frame handling, aero evaluation, actuator-state dynamics. | None found. | Leave untouched. |
| `03_Control/02_Inner_Loop/glider.py` | KEEP_CORE | Frozen geometry, mass/inertia, and control-surface sign source. | None found. | Leave untouched. |
| `03_Control/02_Inner_Loop/implementation_dynamics.py` | DEFER_UNTOUCHED | Hardware/implementation helper outside this cleanup. | None found. | Leave untouched. |
| `03_Control/02_Inner_Loop/implementation_wrappers.py` | DEFER_UNTOUCHED | Hardware/implementation helper outside this cleanup. | None found. | Leave untouched. |
| `03_Control/02_Inner_Loop/linearisation.py` | KEEP_CORE | Canonical state/input names and trim linearisation. | None found. | Leave untouched. |
| `03_Control/02_Inner_Loop/primitives.py` | DEFER_UNTOUCHED | Older inner-loop primitive helper, not part of retired agile branch. | None found. | Leave untouched. |
| `03_Control/02_Inner_Loop/run_linearisation_audit.py` | KEEP_BASELINE | Non-agile audit runner. | None found. | Leave untouched. |
| `03_Control/02_Inner_Loop/run_s4_lite_validation.py` | KEEP_BASELINE | Baseline validation runner. | None found. | Leave untouched. |
| `03_Control/02_Inner_Loop/simulator.py` | DEFER_UNTOUCHED | Older simulator helper outside requested active path. | None found. | Leave untouched. |
| `03_Control/02_Inner_Loop/trim_solver.py` | KEEP_CORE | Trim source used by linearisation and runners. | None found. | Leave untouched. |
| `03_Control/02_Inner_Loop/viability_governor.py` | DEFER_UNTOUCHED | Older inner-loop governor helper, not active retired agile TVLQR. | None found. | Leave untouched. |
| `03_Control/02_Inner_Loop/wind_scenarios.py` | DEFER_UNTOUCHED | Older wind helper outside requested rewrite. | None found. | Leave untouched. |
| `03_Control/03_Primitives/feedback.py` | KEEP_BASELINE | Shared baseline command limiting and attitude hold. | None found. | Leave untouched. |
| `03_Control/03_Primitives/governor.py` | KEEP_CORE | Frozen viability-governor interface. | None found. | Leave untouched. |
| `03_Control/03_Primitives/latency.py` | KEEP_CORE | Frozen calibrated command/latency interface. | None found. | Leave untouched. |
| `03_Control/03_Primitives/metrics.py` | KEEP_BASELINE | Existing metrics schema; may receive only aggressive fields if needed. | Contains old agile naming in helper metadata only, no old module import. | Leave unless aggressive metrics require a minimal addition. |
| `03_Control/03_Primitives/optimise_template.py` | DELETE_RETIRED_AGILE | Old agile template builder coupled to old `TrajectoryPrimitive` and old TVLQR. | `trajectory_primitive`, `tvlqr`, `solve_discrete_tvlqr`, `linearise_trajectory_finite_difference`. | Delete from active path. |
| `03_Control/03_Primitives/primitive.py` | KEEP_CORE | Frozen primitive protocol and entry-condition pattern. | None found. | Leave untouched. |
| `03_Control/03_Primitives/rollout.py` | KEEP_CORE | Frozen rollout interface, RK4, command layer, logging, checks. | None found. | Leave untouched. |
| `03_Control/03_Primitives/run_primitive_audit.py` | KEEP_BASELINE | Non-agile primitive audit runner. | None found. | Leave untouched. |
| `03_Control/03_Primitives/templates.py` | KEEP_BASELINE | Baseline `nominal_glide`, `recovery`, and `mild_bank_reversal_probe`. | None found. | Leave untouched. |
| `03_Control/03_Primitives/trajectory_primitive.py` | DELETE_OLD_TVLQR | Old TVLQR-specific primitive using `TrajectoryPrimitive`, `k_lqr`, and `s_mats`. | Old `TrajectoryPrimitive`, `k_lqr`, `s_mats`. | Delete from active path. |
| `03_Control/03_Primitives/turn_trajectory_optimisation.py` | DELETE_RETIRED_AGILE | Retired regular-turn/OCP30 branch. | `trajectory_primitive`, `tvlqr`, old TVLQR helpers. | Delete from active path. |
| `03_Control/03_Primitives/tvlqr.py` | DELETE_OLD_TVLQR | Old generic/turn TVLQR implementation superseded by new aggressive wrapper. | Defines old `solve_discrete_tvlqr` and `linearise_trajectory_finite_difference`. | Delete from active path. |
| `03_Control/04_Scenarios/arena.py` | KEEP_CORE | Frozen tracker/safety-volume definitions. | None found. | Leave untouched. |
| `03_Control/04_Scenarios/run_agile_feasibility.py` | DELETE_RETIRED_AGILE | Retired agile feasibility runner tied to old `s9_agile` scenarios. | Indirect old agile path via `run_one`/`run_sweep`. | Delete from active path. |
| `03_Control/04_Scenarios/run_agile_template_search.py` | DELETE_RETIRED_AGILE | Retired agile template-search runner. | `optimise_template`. | Delete from active path. |
| `03_Control/04_Scenarios/run_agile_trajectory_optimisation.py` | DELETE_RETIRED_AGILE | Retired OCP30/Phase-2 TVLQR runner. | `trajectory_primitive`, `turn_trajectory_optimisation`, `tvlqr`, `ocp30`, `phase2_tvlqr`. | Delete from active path. |
| `03_Control/04_Scenarios/run_batch.py` | REWRITE_CLEAN | Batch runner imports scenario list that currently includes old agile scenarios. | Indirect via `scenarios.py`. | Minimally rewrite to non-agile scenario list only. |
| `03_Control/04_Scenarios/run_one.py` | REWRITE_CLEAN | Scenario runner materialises old `AgileTurnTemplate` candidates. | `optimise_template`, old agile materialisation. | Remove old agile materialisation while preserving baseline scenarios. |
| `03_Control/04_Scenarios/run_sweep.py` | REWRITE_CLEAN | Entry sweep imports old materialisation from `run_one` and contains old agile gate helper. | Indirect old agile materialisation and old agile naming. | Remove old materialisation dependency and old agile gate helper. |
| `03_Control/04_Scenarios/scenarios.py` | REWRITE_CLEAN | Scenario factory imports old agile template module and exposes old `s9_agile` scenarios. | `optimise_template`, old agile target helpers. | Remove old agile scenarios/imports; add deterministic aggressive entry helper only. |
| `03_Control/04_Scenarios/updraft_models.py` | KEEP_BASELINE | Updraft model loader used by baseline scenarios. | None found. | Leave untouched. |
| `03_Control/05_Results/plot_batch.py` | DEFER_UNTOUCHED | Plotting utility outside requested branch. | None found. | Leave untouched. |
| `03_Control/05_Results/plot_one.py` | DEFER_UNTOUCHED | Plotting utility outside requested branch. | None found. | Leave untouched. |
| `03_Control/05_Results/plot_style.py` | DEFER_UNTOUCHED | Plotting style utility outside requested branch. | None found. | Leave untouched. |
| `03_Control/05_Results/plotting.py` | REWRITE_CLEAN | Plotting utility imported old scenario materialisation after runner cleanup. | Indirect old agile materialisation through `run_one`. | Removed old materialisation import/call and old retired scenario mappings. |

## Post-Cleanup Action Summary

- Deleted old agile/OCP30/TVLQR code from the active Python path: `optimise_template.py`, `trajectory_primitive.py`, `turn_trajectory_optimisation.py`, `tvlqr.py`, old agile scenario runners, and old agile/TVLQR-only tests.
- Deleted retired generated result directories `03_Control/05_Results/03_primitives/05_*` through `12_*` that belonged to the retired agile/TVLQR branch.
- Removed old agile/TVLQR imports from active baseline runner code in `scenarios.py`, `run_one.py`, `run_sweep.py`, and plotting support.
- Preserved frozen plant, glider, linearisation, latency, primitive protocol, rollout, governor, and arena files without edits.
- Added the new aggressive high-incidence rewrite modules and new cleanup guard test after the old TVLQR purge.
