# D2/W2 Readiness Recommendation

Classification: `ready_for_D2_boundary_refinement`.

D1a is sufficient to proceed to D2 boundary refinement. It remains D1a thesis-scale simulation evidence only and does not complete W2, W3, W4, W5, mission, hardware, or sim-to-real validation.

## W1 Target-Ladder Findings
- single_fan_branch bank_yaw_energy_retaining 15 deg dir -1: 115/1471 (0.0782)
- four_fan_branch bank_yaw_energy_retaining 15 deg dir -1: 107/1471 (0.0727)
- four_fan_branch bank_yaw_energy_retaining 15 deg dir 1: 95/1471 (0.0646)
- four_fan_branch wingover_lite 15 deg dir -1: 70/1471 (0.0476)
- single_fan_branch bank_yaw_energy_retaining 15 deg dir 1: 67/1471 (0.0455)
- four_fan_branch wingover_lite 15 deg dir 1: 52/1471 (0.0354)
- four_fan_branch canyon_steep_bank 15 deg dir -1: 51/1471 (0.0347)
- single_fan_branch wingover_lite 15 deg dir -1: 46/1471 (0.0313)
- four_fan_branch canyon_steep_bank 15 deg dir 1: 45/1471 (0.0306)
- single_fan_branch canyon_steep_bank 15 deg dir -1: 38/1471 (0.0258)
- single_fan_branch canyon_steep_bank 15 deg dir 1: 31/1471 (0.0211)
- single_fan_branch wingover_lite 15 deg dir 1: 23/1471 (0.0156)
- Agile targets at 45 deg and above produced 0 W1 successes.

## W0-Failed / W1-Valid Counts
- W0_failed_W1_valid_single_fan: 130
- W0_failed_W1_valid_four_fan: 135

## Recommended Next Run
- Run D2 boundary refinement before W2 complex-updraft replay.
- Use 5,000 to 20,000 extra W1-focused or paired cases per fan-layout branch.
- Concentrate D2 on 15 deg and weak 30 deg agile cases, W1 boundary cells, W0/W1 disagreements, updraft-edge samples, and safety-margin boundaries.
- Prepare W2 with W1 winners, boundary cases, and W0/W1 disagreement cases, but execute W2 only after D2 confirms a stable branch-local shortlist.
- W2 reduced preflight count should be 20,000 to 40,000 complex-updraft cases per fan-layout branch.

## Stop Rules
- Move from D2 to W2 only when branch-local W1 boundary cells are stable enough to choose W2 representatives without cross-layout promotion.
- Move beyond W2 only after nominal-latency W2 candidates preserve safety margins under complex updraft variation.
- Do not escalate to W3/W4/W5 claims without conservative-latency stress, mission evaluation, and real-flight evidence.
