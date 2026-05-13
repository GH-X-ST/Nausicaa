from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from flight_dynamics import adapt_glider
from glider import build_nausicaa_glider
from latency import CommandToSurfaceLayer
from linearisation import linearise_trim
from optimise_template import (
    agile_phase_metadata,
    agile_template_to_dict,
    build_agile_reversal_candidate,
    load_selected_agile_template,
)
from primitive import build_primitive_context
from run_agile_template_search import FAMILIES, TARGETS_DEG, agile_search_templates


def test_agile_search_template_grid_is_compact_and_explicit() -> None:
    templates = agile_search_templates()

    assert sorted({float(template.target_heading_deg) for template in templates}) == list(
        TARGETS_DEG
    )
    assert sorted({template.family for template in templates}) == sorted(FAMILIES)
    assert "updraft_assisted" not in {template.family for template in templates}
    assert len(templates) == len(TARGETS_DEG) * len(FAMILIES)


def test_agile_phase_metadata_has_complete_horizon_keys() -> None:
    template = agile_search_templates(targets_deg=(30.0,))[1]
    phases = agile_phase_metadata(template)

    assert tuple(phases) == (
        "entry",
        "brake_or_pitch",
        "roll_yaw_redirect",
        "turn_hold_or_heading_capture",
        "recover",
        "exit_check",
    )
    assert all(float(phase["duration_s"]) >= 0.0 for phase in phases.values())
    assert float(phases["exit_check"]["start_s"]) == float(phases["recover"]["end_s"])


def test_selected_agile_template_loads_from_manifest(tmp_path: Path) -> None:
    template = agile_search_templates(targets_deg=(30.0,))[1]
    manifest = {
        "selected_templates": {
            "030": agile_template_to_dict(template),
        }
    }
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    (manifest_dir / "agile_template_search_seed1.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    loaded = load_selected_agile_template(
        repo_root=Path.cwd(),
        seed=1,
        target_heading_deg=30.0,
        search_root=tmp_path,
    )

    assert loaded == template


def test_agile_candidate_trajectory_arrays_are_finite() -> None:
    aircraft = adapt_glider(build_nausicaa_glider())
    linear_model = linearise_trim(aircraft=aircraft)
    context = build_primitive_context(linear_model.x_trim, linear_model.u_trim)
    template = agile_search_templates(targets_deg=(30.0,))[1]

    primitive = build_agile_reversal_candidate(
        template=template,
        x0=linear_model.x_trim,
        context=context,
        aircraft=aircraft,
        wind_model=None,
        wind_mode="none",
        command_layer=CommandToSurfaceLayer(),
    )

    for value in (
        primitive.times_s,
        primitive.x_ref,
        primitive.u_ff,
        primitive.a_mats,
        primitive.b_mats,
        primitive.k_lqr,
        primitive.s_mats,
    ):
        assert value is not None
        assert np.all(np.isfinite(value))
    assert primitive.metadata["primitive_family"] == "brake_roll_yaw_recovery"
    assert primitive.metadata["candidate_id"] == "030_a"
