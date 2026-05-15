# Phase 2 TVLQR OCP30 Boundary Report

The W0 30 deg OCP candidate was not promoted beyond Phase 2.

- phase 2 status: `boundary_only`
- active failure class: `latency_limited_high_alpha`
- all failure classes: `latency_limited_high_alpha;terminal_recovery_limited`
- limitation: `angle of attack exceeded bound`
- hard 30 deg OCP reproduced: `True`
- open-loop no-latency gate: `True`
- closed-loop no-latency gate: `True`
- nominal-latency gate: `False`
- terminal recovery sensitivity gate: `False`

No physical sign, command-order, state-order, arena-bound, or command-authority changes were made to force promotion.

## Best 30 Deg OCP Row

- candidate variant: `baseline`
- label: `accepted_low_alpha`
- failure reason: ``
- directed heading change deg: `28.293163149683984`
- heading threshold deg: `24.0`
- min wall distance m: `0.25`
- terminal altitude m: `2.5263011311490597`
- terminal alpha deg: `4.8362953644871185`
- terminal beta deg: `25.143698311128077`
- terminal rate norm rad/s: `1.2289568285903856`
- max alpha deg: `4.8362953644871185`
- saturation fraction: `0.0`
- slack max: `0.0`
