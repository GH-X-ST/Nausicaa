# 03_Control Reset Inventory

Date: 2026-05-15

## Reset Summary

`03_Control` was reset to a small model, updraft, arena, latency-facts, and plotting-style foundation. No controller, primitive rollout, OCP, TVLQR, governor, mission, library, factory, or scenario-runner implementation was added or retained.

No backup or archive directory was created. Existing generated results and old CODEX audit/inspection outputs were not preserved.

## Final Active Tree

```text
03_Control/
  02_Inner_Loop/
    A_model_parameters/
      build_mass_properties_estimate.py
      mass_properties_estimate.py
    flight_dynamics.py
    glider.py
    linearisation.py
    trim_solver.py
  03_Primitives/
    latency.py
  04_Scenarios/
    arena.py
    updraft_models.py
  05_Results/
    plot_style.py
```

## Kept Files

| Path | Category | Reason | Imports checked |
| --- | --- | --- | --- |
| `03_Control/02_Inner_Loop/glider.py` | Aircraft model | Canonical geometry, mass, inertia, surfaces, and sign conventions for the retained glider model. | Yes: compile, retained import check, targeted tests. |
| `03_Control/02_Inner_Loop/flight_dynamics.py` | Aircraft model | Retained nonlinear model and wind/frame conventions required for trim and linearisation. | Yes: compile, retained import check, targeted tests. |
| `03_Control/02_Inner_Loop/trim_solver.py` | Model utility | Retained compact trim helper for the foundation model only. | Yes: compile, retained import check, targeted tests. |
| `03_Control/02_Inner_Loop/linearisation.py` | Model utility | Retained state/input ordering and finite-difference model linearisation. | Yes: compile, retained import check, targeted tests. |
| `03_Control/02_Inner_Loop/A_model_parameters/build_mass_properties_estimate.py` | Aircraft model parameters | Retained deterministic mass-property generation helper. | Yes: optional compile. |
| `03_Control/02_Inner_Loop/A_model_parameters/mass_properties_estimate.py` | Aircraft model parameters | Retained SI mass-property estimate constants. | Yes: optional compile. |
| `03_Control/03_Primitives/latency.py` | Measured constants | Retained command lattice, surface limits, aggregate limits, latency envelope constants, and stateless conversion/quantisation helpers. Stateful `CommandToSurfaceLayer` FIFO logic was removed. | Yes: compile, retained import check, targeted tests. |
| `03_Control/04_Scenarios/arena.py` | Measured constants | Retained true/tracker safety bounds and arena margin helpers. | Yes: compile, retained import check, targeted tests. |
| `03_Control/04_Scenarios/updraft_models.py` | Updraft model | Retained measured/fitted updraft loaders, randomisation helpers, and wind-field smoke-test functionality. | Yes: compile, retained import check, targeted tests. |
| `03_Control/05_Results/plot_style.py` | Plotting template | Retained only the general plotting style/template support for later result generation. | Yes: compile, retained import check. |

## Deleted Categories

| Area | Deleted scope |
| --- | --- |
| `02_Inner_Loop` duplicate model paths | Old duplicate/compatibility model wrappers, legacy simulator paths, old implementation wrappers, primitive shims, wind scenario shims, viability governor code, old runners, audit writers, and the old PDF datasheet after inspection showed it was general notation rather than unique measured constants. |
| `03_Primitives` controllers and rollouts | All primitive, controller, governor, rollout, metrics, aggressive-reversal, OCP, and TVLQR implementation files except `latency.py`. |
| `04_Scenarios` runners and factories | All scenario runners, launch-state contracts, factories, and metrics writers except `arena.py` and `updraft_models.py`. |
| `05_Results` generated outputs | All generated results, reports, figures, traces, old plotting runners, and old result subtrees. Only `plot_style.py` remains. |
| Tests importing removed APIs | Tests that imported deleted modules or the removed stateful latency layer API were deleted. Retained tests now cover only model, arena, updraft, and latency-constant behavior plus unrelated tests outside this reset scope. |

## Validation

### Required compile check

Command:

```powershell
python -m py_compile 03_Control/02_Inner_Loop/glider.py 03_Control/02_Inner_Loop/flight_dynamics.py 03_Control/02_Inner_Loop/trim_solver.py 03_Control/02_Inner_Loop/linearisation.py 03_Control/04_Scenarios/updraft_models.py 03_Control/03_Primitives/latency.py 03_Control/04_Scenarios/arena.py 03_Control/05_Results/plot_style.py
```

Output:

```text
<no output; exit code 0>
```

Additional optional compile check:

```powershell
python -m py_compile 03_Control/02_Inner_Loop/A_model_parameters/build_mass_properties_estimate.py 03_Control/02_Inner_Loop/A_model_parameters/mass_properties_estimate.py
```

Output:

```text
<no output; exit code 0>
```

### Retained import check

Command: PowerShell here-string import check with `PYTHONDONTWRITEBYTECODE=1`.

Output:

```text
retained imports ok
```

### Old control token scan

Scope: active `03_Control` files only. The reset inventory is intentionally outside `03_Control`.

Output:

```text
old control tokens absent from active 03_Control
```

### 05_Results scan

Output:

```text
05_Results contains only plotting template/style files
```

### Targeted retained pytest suite

Command:

```powershell
python -m pytest -q tests/test_state_order.py tests/test_control_signs.py tests/test_trim_residual.py tests/test_linearisation_finite_difference.py tests/test_panel_cg_uniform_wind.py tests/test_arena_bounds.py tests/test_full_range_command_policy.py tests/test_latency_step_response.py tests/test_surface_limits.py tests/test_surface_quantisation.py tests/test_updraft_model_shapes.py tests/test_updraft_model_reproduction.py tests/test_updraft_randomisation.py
```

Output:

```text
25 passed in 11.38s
```

Note: standalone `pytest` was not available on `PATH`, so the targeted suite was run with `python -m pytest`.

### Final active tree listing

Output:

```text
03_Control\02_Inner_Loop\A_model_parameters\build_mass_properties_estimate.py
03_Control\02_Inner_Loop\A_model_parameters\mass_properties_estimate.py
03_Control\02_Inner_Loop\flight_dynamics.py
03_Control\02_Inner_Loop\glider.py
03_Control\02_Inner_Loop\linearisation.py
03_Control\02_Inner_Loop\trim_solver.py
03_Control\03_Primitives\latency.py
03_Control\04_Scenarios\arena.py
03_Control\04_Scenarios\updraft_models.py
03_Control\05_Results\plot_style.py
```

## Known Limitations

- Full `pytest` was not used as an acceptance gate because the worktree already had an unrelated deleted packaging backend: `nausicaa_build_backend.py`.
- Pre-existing dirty worktree deletions under `docs/control/*` and `nausicaa_build_backend.py` were intentionally left untouched.
- `updraft_models.py` was retained otherwise unchanged because it has no controller or scenario-runner imports. Its labelled analytic debug proxy is kept only as a wind-field/randomisation smoke-test helper.
- Validation generated Python bytecode caches during compile/import/test runs; those generated `__pycache__` directories were removed after validation to keep the active tree clean.
