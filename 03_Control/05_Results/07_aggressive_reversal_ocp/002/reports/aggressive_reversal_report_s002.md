# Guided Aggressive Reversal OCP Report

This is W0/no-wind aggressive-reversal evidence only. It preserves the
existing plant, state order, command bridge, safety volume, and surface
limits. It is not high-incidence validation, not real-flight validation,
not TVLQR, not governor, and not outer-loop evidence.

## Summary

- Overall status: `boundary_evidence`
- Largest finite target: `30.0` deg
- Largest recoverable target: `0.0` deg
- Largest successful target: `0.0` deg
- Replay defect tolerance: `1e-05`

## Target Outcomes

| Target deg | Method | Family | OCP attempted | OCP converged | Replay finite | Recoverable | Success | Failure label | Limiter |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30.0 | phase_search | long_perch_slow_redirect | True | False | True | False | False | speed_low | high_alpha_drag_limited |

## 30 Deg Physics-First Audit

- Manoeuvre classification: `speed-collapse pitch-up redirect`
- Shape class: `speed_collapse_pitch_redirect`
- Active tradeoff: `high_alpha_drag_limited`
- Strict W0 primitive success: `False`
- Relaxed recovery-required evidence: `False`
- Updraft-assisted boundary evidence: `False`
- Unload/exit descent: `-0.008318989208518701` m
- Unload/exit speed gain: `-0.9386690594280405` m/s
- Specific energy lost: `1.6637624523809689` m
- Ideal descent to regain 5 m/s: `1.0759461283078506` m

Run 002 does not attempt targets larger than 30 deg when invoked with
`--targets 30`; larger targets require a later pass after this
physics-first classification is reviewed.

## OCP Diagnostics

| Target deg | nlp_constructed | ipopt_called | direct_ocp_attempted | direct_ocp_converged | solver_status | constraint_residual_max | replay_defect_max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 30.0 | True | True | True | False | solver_exception:RuntimeError | nan | nan |

## Manoeuvre-Family Guidance

Every target is seeded from the fixed family inventory: `short_perch_yaw_redirect, long_perch_slow_redirect, roll_dominant_banked_redirect, split_pulse_redirect, early_unload_descend_capture, dive_perch_redirect_30, reduced_perch_redirect_30, bank_yaw_redirect_30, early_unload_recovery_30`.
Failure labels map deterministically to the next family or limiter; no
unconstrained smooth-turn-only path is used.

## No-Overclaiming Flags

- OCP implemented: `True`
- TVLQR implemented: `False`
- Governor implemented: `False`
- Outer loop implemented: `False`
- High-incidence validation claim: `False`
- Raw normalised commands enter state derivative: `False`