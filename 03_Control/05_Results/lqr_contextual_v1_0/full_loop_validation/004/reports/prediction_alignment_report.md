# v4.9 Paired Full-Loop Validation Report

- Status: `complete`
- Episodes: `600`
- Primitive executions: `606`
- Terminal-useful rate: `0.170000`
- Hard-failure rate: `0.610000`
- Paired memory effect label: `mixed_memory_effect`
- Policies are compared on common-random paired launch/environment episodes.
- Episodic memory policies persist belief across launches.
- X/y terminal-useful evidence is terminal episode evidence, not continuation success.
- No controller retuning or identity mutation is performed.
- Claim boundary: simulation-only paired full-loop validation.
