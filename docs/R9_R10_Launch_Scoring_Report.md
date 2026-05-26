# R9/R10 Launch Score Audit Metric

## Scope

This report defines a simple audit score for R9 fixed-case repeated-launch validation and R10 environment-only changed-case validation. The score is secondary evidence only. It does not replace strict pass/fail gates for pairing, launch counts, hard-failure rate, no-viable rate, floor/ceiling violations, sequence compliance, or full R9/R10 validation completion.

No hardware readiness, real-flight transfer, mission success, full autonomy, or memory-improvement claim is made by this score alone.

This score also cannot override W3 launch-capture survival gates. If launch_capable variants are only downgraded in W3, as in W3 run 015 (`0 survived / 192 downgraded`), the score may be used to diagnose failure modes but must not be used to claim post-W3 launch-aware readiness, R9 pass, R10 pass, or memory improvement.

Current stage placement: the score is applied only after robust randomized R5 W0/W1 synthesis has produced a frozen controller bundle and frozen held-out W3 validation has accepted launch-capable, inflight, and recovery/safe-exit evidence. Optional W2 diagnostics cannot make this score pass an R9/R10 gate.

## Per-Launch Score

Each final held-out launch receives:

```text
launch_score =
  base_failure_penalty
  + 100
    * M_outcome
    * M_safety
    * M_viability
    * F_net_energy
    * F_energy_loss
    * F_flight_time
    * F_wall_margin
```

The score is designed so failures can become negative instead of merely becoming zero.

## Specific Energy

Energy is scored as specific mechanical energy in metres:

```text
E = z_w + V^2 / (2g)
V = sqrt(u^2 + v^2 + w^2)
g = 9.80665 m/s^2
```

For each selected primitive:

```text
delta_E_i = E_end_i - E_start_i
```

Episode-level energy metrics:

```text
net_specific_energy_delta_m =
  E_final - E_initial

gross_specific_energy_gain_m =
  sum(max(delta_E_i, 0))

gross_specific_energy_loss_m =
  sum(max(-delta_E_i, 0))
```

Score factors:

```text
F_net_energy =
  clip(1.0 + net_specific_energy_delta_m / 2.0, 0.25, 1.75)

F_energy_loss =
  clip(1.0 - gross_specific_energy_loss_m / 2.0, 0.50, 1.00)
```

This separates useful final energy gain from energy wasted during manoeuvres.

## Flight Time

Flight time uses primitive count, not lift dwell:

```text
episode_flight_time_s =
  selected_primitive_step_count * 0.100

target_episode_time_s =
  1.5

F_flight_time =
  clip(episode_flight_time_s / target_episode_time_s, 0.10, 1.25)
```

The 1.5 s target corresponds to 15 primitives. Shorter four-primitive runs are still valid for gate checks but receive a lower time factor.

## Wall Margin

The arena is narrow, so close-but-inside wall flight is not heavily penalised:

```text
F_wall_margin =
  0.50 if min_wall_margin_m < 0.00
  0.80 if 0.00 <= min_wall_margin_m < 0.05
  0.95 if 0.05 <= min_wall_margin_m < 0.15
  1.00 if 0.15 <= min_wall_margin_m < 0.40
  1.05 if min_wall_margin_m >= 0.40
```

Boundary violations remain handled by the safety multiplier and failure penalty.

## Outcome Multiplier

```text
M_outcome =
  1.00 if safe_success and terminal_useful
  0.90 if safe_success and lift_capture
  0.75 if safe_success
  0.50 if terminal_useful and not hard_failure
  0.35 if lift_capture and not hard_failure
  0.10 otherwise
```

Definitions:

- `safe_success`: no hard failure and at least one continuation-valid or terminal-useful primitive.
- `terminal_useful`: the episode reaches a useful terminal/safe-exit/recovery handoff state.
- `lift_capture`: the episode spends nonzero time in useful lift.

## Safety Multiplier

```text
M_safety =
  0.00 if hard_failure
  0.00 if floor_or_ceiling_violation
  0.60 if wall boundary issue but not hard failure
  1.00 otherwise
```

## Viability Multiplier

```text
M_viability =
  0.00 if no_viable_primitive at launch
  0.50 if no_viable_primitive after at least one primitive
  1.00 otherwise
```

## Failure Penalty

```text
base_failure_penalty =
  -100 if hard_failure
  -100 if floor_or_ceiling_violation
  -70  if no_viable_primitive at launch
  -40  if no_viable_primitive after at least one primitive
  -30  if wall boundary issue but not hard failure
  0    otherwise
```

Only the most severe applicable penalty is applied.

## Paired Memory Comparison

Memory and safe-explore policies are compared only by controlled final-launch pairing.

Memory policies compare against `no_memory_baseline`:

```text
paired_delta_launch_score =
  launch_score(memory_policy, same paired final launch)
  - launch_score(no_memory_baseline, same paired final launch)
```

Safe-explore policies compare against matching directional-memory history:

```text
safe_explore_then_exploit_hN
minus
directional_3d_residual_memory_hN
```

Summary statistics:

```text
mean_paired_delta_launch_score
median_paired_delta_launch_score
win_rate
loss_rate
safety_regression_rate
memory_changed_selection_rate
exploration_changed_selection_rate
mean_net_specific_energy_delta_m_delta
mean_gross_specific_energy_loss_m_delta
mean_episode_flight_time_s_delta
```

## Output Tables

R9 and R10 post analysis writes:

```text
metrics/final_launch_score.csv
metrics/paired_memory_score_delta.csv
metrics/paired_safe_explore_score_delta.csv
metrics/paired_score_delta_summary.csv
```

The existing `policy_history_comparison.csv`, `library_size_case_comparison.csv`, and R10 `environment_block_comparison.csv` also include launch-score summaries.
