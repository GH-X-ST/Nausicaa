# R2-R5 Contextual Primitive Restart Build Note

## Status

This pass adds the first environment-conditioned primitive interfaces after the model-only reset:

- environment context rows;
- active primitive catalogue;
- smoke rollout evidence rows;
- chunked contextual archive preflight scaffold.

The selected evidence policy is temp-only. Tests write smoke/preflight outputs under pytest temporary directories. The active result root remains `03_Control/05_Results/.gitkeep` only.

## Git State At Start Of This Pass

`git status --short` showed:

```text
 M 03_Control/tests/test_repo_housekeeping.py
 D "docs/MATLAB Coding.txt"
 D "docs/Python Plotting Guidance.txt"
?? 03_Control/03_Primitives/prim_cat.py
?? 03_Control/03_Primitives/prim_roll.py
?? 03_Control/04_Scenarios/env_ctx.py
?? 03_Control/04_Scenarios/run_ctx_archive.py
?? 03_Control/tests/test_ctx_archive_smoke.py
?? 03_Control/tests/test_env_ctx.py
?? 03_Control/tests/test_prim_cat.py
?? 03_Control/tests/test_prim_roll.py
```

Previous reset notes recorded local Git metadata ACL issues that blocked branch creation and staging. This note does not imply that a clean committed branch exists.

## Evidence Boundary

The generated rows are schema, runtime, and smoke evidence only. They are not controller performance evidence and do not support real-flight transfer, hardware readiness, mission success, full robustness, or environment-generalisation claims.

## Contamination Status

The new implementation uses state, local flow context, primitive definitions, smoke rollout rows, and chunked table storage. Retired archive, chain, package, and hardware-result workflows are not imported.

## Validation

- Focused R2-R5 tests: `14 passed`.
- Full retained test suite: `101 passed`.
- Python compile check over retained and new modules: passed.
- `git diff --check`: passed with line-ending warning only.
- Result-root audit: `03_Control/05_Results` contains only `.gitkeep`.
- File-size audit: no non-Git file above 100 MB.

Smoke/preflight files were generated only under pytest temporary directories during validation and are not retained as project evidence.
