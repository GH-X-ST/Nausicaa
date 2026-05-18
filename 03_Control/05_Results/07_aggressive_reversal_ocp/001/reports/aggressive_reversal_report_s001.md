# Guided Aggressive Reversal OCP Report

This is W0/no-wind aggressive-reversal evidence only. It preserves the
existing plant, state order, command bridge, safety volume, and surface
limits. It is not high-incidence validation, not real-flight validation,
not TVLQR, not governor, and not outer-loop evidence.

## Summary

- Overall status: `boundary_evidence`
- Largest finite target: `180.0` deg
- Largest recoverable target: `0.0` deg
- Largest successful target: `0.0` deg
- Replay defect tolerance: `1e-05`

## Target Outcomes

| Target deg | Method | Family | OCP attempted | OCP converged | Replay finite | Recoverable | Success | Failure label | Limiter |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 15.0 | phase_search | long_perch_slow_redirect | True | False | True | False | False | speed_low | physical_boundary |
| 30.0 | phase_search | long_perch_slow_redirect | True | False | True | False | False | speed_low | physical_boundary |
| 60.0 | phase_search | short_perch_yaw_redirect | True | False | True | False | False | under_turning | insufficient_manoeuvre_seed |
| 90.0 | phase_search | short_perch_yaw_redirect | True | False | True | False | False | under_turning | insufficient_manoeuvre_seed |
| 120.0 | phase_search | early_unload_descend_capture | True | False | True | False | False | under_turning | insufficient_manoeuvre_seed |
| 180.0 | phase_search | early_unload_descend_capture | True | False | True | False | False | true_safety_violation | physical_safety_boundary |

## OCP Diagnostics

| Target deg | nlp_constructed | ipopt_called | direct_ocp_attempted | direct_ocp_converged | solver_status | constraint_residual_max | replay_defect_max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 15.0 | True | True | True | False | not_run | nan | nan |
| 30.0 | True | True | True | False | not_run | nan | nan |
| 60.0 | True | True | True | False | not_run | nan | nan |
| 90.0 | True | True | True | False | not_run | nan | nan |
| 120.0 | True | True | True | False | not_run | nan | nan |
| 180.0 | True | True | True | False | not_run | nan | nan |

## Manoeuvre-Family Guidance

Every target is seeded from the fixed family inventory: `short_perch_yaw_redirect, long_perch_slow_redirect, roll_dominant_banked_redirect, split_pulse_redirect, early_unload_descend_capture`.
Failure labels map deterministically to the next family or limiter; no
unconstrained smooth-turn-only path is used.

## No-Overclaiming Flags

- OCP implemented: `True`
- TVLQR implemented: `False`
- Governor implemented: `False`
- Outer loop implemented: `False`
- High-incidence validation claim: `False`
- Raw normalised commands enter state derivative: `False`