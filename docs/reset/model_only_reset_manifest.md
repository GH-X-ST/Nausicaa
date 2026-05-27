# Model-Only Contextual Primitive Reset Manifest

Status: historical reset manifest. This file is retained under `docs/reset/`
as deleted-history context only. It is not an active project plan, allowlist, or
stage contract. For current method control, use
`docs/Glider_Control_Project_Plan.md`.

## Pre-Reset State

- Pre-reset HEAD: `98595797abdf23b257f1a884504fca59436997ca`
- Starting branch: `main`
- Requested reset branch: `model-only-contextual-primitives-reset`
- Branch creation status: attempted before deletions, but Git could not create `.git/refs/heads/model-only-contextual-primitives-reset.lock` because the workspace denied write permission under `.git/refs/heads`. The reset changes therefore remain in the current working tree and this failure is part of the audit record.

Pre-reset tracked status:

```text
 D "docs/Python Coding to CODEX.txt"
 D docs/Skills.txt
 D docs/control/03_control_reset_inventory.md
 D docs/control/aerosandbox_envelope_check_protocol.md
 D docs/control/longitudinal_moment_audit_protocol.md
 M docs/housekeeping_and_naming_rules.md
```

Pre-reset ignored/untracked highlights:

```text
docs/Glider_Control_Project_Plan.md
docs/Python Coding Instruction.txt
docs/Skills.md
03_Control/05_Results/** generated result roots
__pycache__ and .pytest_cache folders
```

## Allowlist

The active allowlist is intentionally small:

```text
03_Control/02_Inner_Loop/glider.py
03_Control/02_Inner_Loop/flight_dynamics.py
03_Control/02_Inner_Loop/linearisation.py
03_Control/02_Inner_Loop/trim_solver.py
03_Control/02_Inner_Loop/A_model_parameters/mass_properties_estimate.py
03_Control/03_Primitives/state_contract.py
03_Control/03_Primitives/command_contract.py
03_Control/03_Primitives/metric_contract.py
03_Control/03_Primitives/latency.py
03_Control/03_Primitives/wing_wind_descriptors.py
03_Control/03_Primitives/dense_archive_runtime.py
03_Control/03_Primitives/dense_archive_table_io.py
03_Control/03_Primitives/dense_archive_chunking.py
03_Control/04_Scenarios/arena.py
03_Control/04_Scenarios/arena_contract.py
03_Control/04_Scenarios/scenario_contract.py
03_Control/04_Scenarios/updraft_models.py
03_Control/tests/* retained foundation tests only
docs/Glider_Control_Project_Plan.md
docs/Skills.md
docs/Python Coding Instruction.txt
docs/housekeeping_and_naming_rules.md
docs/Daily_Schedule.txt
docs/abbr.md
docs/reset/*
README.md
LICENSE
pyproject.toml
requirements.txt
.gitignore
```

Required data inputs:

```text
02_Glider_Design/C_results/nausicaa_results.csv
01_Thermal/S01.xlsx
01_Thermal/S02.xlsx
01_Thermal/B_results/single_var_params.xlsx
01_Thermal/B_results/four_var_params.xlsx
01_Thermal/B_results/Single_Fan_Annular_GP/single_annular_gp_grid_predictions.xlsx
01_Thermal/B_results/Four_Fan_Annular_GP/four_annular_gp_grid_predictions.xlsx
```

## Files Removed By Category

- Old fixed-gate and paired archive runners.
- Old branch-specific archive planning, aggregation, profiling, and proof scripts.
- Old reachable-state extraction paths.
- Old primitive-library, old candidate-package, and old selection/governor implementation paths.
- Old hardware-aware replay and environment-randomised replay paths tied to selected archive rows.
- Old agile-turn expansion paths and tests.
- Old generated evidence under `03_Control/05_Results/**`.
- Old active tests whose imports require deleted implementation paths.
- Stale Codex/internal reset plans outside `docs/reset/`.

## Files Kept By Category

- Glider and flight-dynamics model foundation.
- Timing and command contract foundation.
- Measured/fitted updraft model loader and wing-scale local-flow descriptors.
- Contextual runtime/storage utilities only after neutralising old branch/package vocabulary.
- Active contract documents replaced unconditionally with environment-conditioned versions.
- Foundation tests and contamination/housekeeping tests.

## Claim Boundary

This reset produces no controller result, primitive policy, transfer claim, hardware-readiness claim, W-stage completion claim, or environment-generalisation claim.

The only valid claim after completion is:

```text
the repository has been reduced to a clean modelling and runtime foundation for environment-conditioned primitive development
```

## Planned Validation

```text
python -m py_compile <retained importable modules>
python -m pytest -q <retained foundation tests>
git diff --check
file-size audit: list tracked files above 50 MB and fail non-approved files above 100 MB
result audit: fail if 03_Control/05_Results contains anything except .gitkeep
contamination audit: fail if active retained files contain forbidden old-method traces
```
