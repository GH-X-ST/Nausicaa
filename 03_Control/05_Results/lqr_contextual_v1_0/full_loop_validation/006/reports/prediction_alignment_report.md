# v4.9 Paired Full-Loop Validation Report

- Status: `complete`
- Episodes: `1800`
- Primitive executions: `1836`
- Terminal-useful rate: `0.186667`
- Hard-failure rate: `0.553333`
- Paired memory effect label: `memory_benefit_mixed`
- Policies are compared on common-random paired launch/environment episodes.
- Episodic memory policies persist belief across launches.
- X/y terminal-useful evidence is terminal episode evidence, not continuation success.
- No controller retuning or identity mutation is performed.
- Claim boundary: simulation-only paired full-loop validation.
