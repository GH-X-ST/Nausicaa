from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CONTROL_ROOT = Path(__file__).resolve().parents[1]
for rel in ("02_Inner_Loop", "03_Primitives", "04_Scenarios"):
    path = CONTROL_ROOT / rel
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dense_archive_table_io import filesystem_path  # noqa: E402
from directional_residual_lift_belief import (  # noqa: E402
    DirectionalResidualObservation,
    belief_snapshot_row,
    initial_directional_residual_lift_belief,
    query_directional_residual_lift_features,
    update_directional_residual_lift_belief,
)
from env_ctx import build_environment_context  # noqa: E402
from env_instance import (  # noqa: E402
    environment_instance_for_mode,
    environment_metadata_from_instance,
)
from env_surrogate import (  # noqa: E402
    READY_STATUS,
    resolve_surrogate_binding,
    wind_field_for_binding,
)
from lqr_controller import lqr_controller_for_primitive_id  # noqa: E402
from prim_cat import active_primitive_catalogue, primitive_by_id  # noqa: E402
from prim_features import primitive_feature_record, primitive_feature_row  # noqa: E402
from prim_model import fit_primitive_outcome_model  # noqa: E402
from prim_roll import (  # noqa: E402
    RolloutConfig,
    blocked_rollout_evidence,
    rollout_with_context_row,
    simulate_primitive_rollout,
)
from prim_select import primitive_selection_row, select_primitive  # noqa: E402
from state_sampling import archive_state_sample_for_row, archive_state_sample_row  # noqa: E402


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Episode smoke config and CLI
# 2) Episode execution
# 3) Output helpers
# =============================================================================


# =============================================================================
# 1) Episode Smoke Config and CLI
# =============================================================================
@dataclass(frozen=True)
class EpisodeSmokeConfig:
    run_id: int
    episode_count: int
    seed: int
    governor_mode: str
    output_root: Path


def parse_args(argv: list[str] | None = None) -> EpisodeSmokeConfig:
    parser = argparse.ArgumentParser(
        description="Run temp-only contextual primitive episode smoke checks."
    )
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--episode-count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--governor-mode",
        choices=("continuation", "terminal_episode"),
        default="terminal_episode",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    return EpisodeSmokeConfig(
        run_id=int(args.run_id),
        episode_count=int(args.episode_count),
        seed=int(args.seed),
        governor_mode=str(args.governor_mode),
        output_root=Path(args.output_root),
    )


# =============================================================================
# 2) Episode Execution
# =============================================================================
def run_contextual_episode_smoke(config: EpisodeSmokeConfig) -> dict[str, object]:
    """Run a small repeated-launch smoke chain without performance claims."""

    if config.episode_count <= 0:
        raise ValueError("episode_count must be positive.")
    if config.governor_mode not in {"continuation", "terminal_episode"}:
        raise ValueError("governor_mode must be continuation or terminal_episode.")

    run_root = Path(config.output_root) / f"episode_smoke_{config.run_id:03d}"
    filesystem_path(run_root / "manifests").mkdir(parents=True, exist_ok=True)
    filesystem_path(run_root / "tables").mkdir(parents=True, exist_ok=True)
    filesystem_path(run_root / "reports").mkdir(parents=True, exist_ok=True)

    primitives = active_primitive_catalogue()
    training_rows: list[dict[str, object]] = []
    episode_rows: list[dict[str, object]] = []
    selector_rows: list[dict[str, object]] = []
    belief = initial_directional_residual_lift_belief()
    belief_rows = [belief_snapshot_row(belief, label="initial")]

    for episode_index in range(int(config.episode_count)):
        row_index = int(config.seed + episode_index)
        w_layer = "W0" if episode_index % 2 == 0 else "W1"
        environment_mode = "dry_air" if w_layer == "W0" else "gaussian_single"
        state_sample = archive_state_sample_for_row(
            row_index,
            seed=config.seed,
            W_layer=w_layer,
            environment_mode=environment_mode,
        )
        state = state_sample.state_vector
        instance = environment_instance_for_mode(
            w_layer,
            environment_mode,
            config.seed + episode_index,
        )
        metadata = environment_metadata_from_instance(instance)
        binding = resolve_surrogate_binding(
            w_layer,
            metadata,
            randomisation_seed=config.seed + episode_index,
        )
        wind = wind_field_for_binding(binding)
        context = build_environment_context(
            state,
            wind_field=wind,
            metadata=metadata,
            latency_case="none" if w_layer == "W0" else "nominal",
            actuator_case="nominal",
            surrogate_binding=binding,
        )
        belief_before = _query_directional_belief(state, belief)
        if len(training_rows) < len(primitives):
            training_rows.extend(
                _bootstrap_lqr_rows(
                    state=state,
                    context=context,
                    primitives=primitives,
                    config=RolloutConfig(
                        W_layer=w_layer,
                        rollout_backend="model_backed_lqr",
                        wind_mode=binding.wind_mode,
                    ),
                    wind_field=wind,
                    episode_index=episode_index,
                    state_sample=state_sample,
                )
            )

        model = fit_primitive_outcome_model(training_rows, k_neighbours=3)
        selection = select_primitive(
            context=context,
            model=model,
            catalogue=primitives,
            governor_mode=config.governor_mode,
            max_uncertainty=1_000_000.0,
        )
        rollout_config = RolloutConfig(
            W_layer=w_layer,
            rollout_backend="model_backed_lqr",
            wind_mode=binding.wind_mode,
        )
        rollout_id = f"episode_{config.run_id:03d}_{episode_index:03d}"
        if not selection.selected_primitive_id:
            primitive = primitive_by_id("safe_exit_or_recovery_handoff")
            evidence = blocked_rollout_evidence(
                rollout_id=rollout_id,
                episode_id=rollout_id,
                initial_state=state,
                context=context,
                primitive=primitive,
                config=rollout_config,
                failure_label="selector_blocked_no_viable_lqr_primitive",
                termination_cause="controller_blocked",
                controller_selection_status="missing_primitive_variant_registry",
            )
            row = rollout_with_context_row(evidence, context)
            row.update(
                {
                    "episode_index": episode_index,
                    "selector_governor_mode": selection.governor_mode,
                    "selector_status": selection.decision_status,
                    "selected_primitive_id": selection.selected_primitive_id,
                    "memory_label": "directional_residual_lift_belief_smoke_no_improvement_claim",
                    "belief_before_local_lift_residual_m_s": belief_before["belief_local_lift_residual_m_s"],
                    "belief_before_uncertainty": belief_before["belief_uncertainty"],
                    "belief_after_local_lift_residual_m_s": belief_before["belief_local_lift_residual_m_s"],
                    "belief_after_uncertainty": belief_before["belief_uncertainty"],
                }
            )
            row.update(archive_state_sample_row(state_sample))
            training_rows.append(row)
            episode_rows.append(row)
            selector_row = primitive_selection_row(selection)
            selector_row.update(archive_state_sample_row(state_sample))
            selector_rows.append(selector_row)
            belief_rows.append(belief_snapshot_row(belief, label=f"episode_{episode_index:03d}"))
            continue
        primitive = primitive_by_id(selection.selected_primitive_id)
        chosen_controller = (
            lqr_controller_for_primitive_id(selection.selected_primitive_id)
            if selection.selected_primitive_id
            else None
        )
        if binding.surrogate_binding_status != READY_STATUS:
            evidence = blocked_rollout_evidence(
                rollout_id=rollout_id,
                episode_id=rollout_id,
                initial_state=state,
                context=context,
                primitive=primitive,
                config=rollout_config,
                failure_label="surrogate_binding_blocked",
                termination_cause="surrogate_binding_blocked",
                controller=chosen_controller,
                controller_selection_status="W01_variant_registry_candidate",
            )
        else:
            evidence = simulate_primitive_rollout(
                rollout_id=rollout_id,
                episode_id=rollout_id,
                initial_state=state,
                context=context,
                primitive=primitive,
                config=rollout_config,
                wind_field=wind,
                controller=chosen_controller,
                controller_selection_status="W01_variant_registry_candidate",
            )
        row = rollout_with_context_row(evidence, context)
        row.update(
            {
                "episode_index": episode_index,
                "selector_governor_mode": selection.governor_mode,
                "selector_status": selection.decision_status,
                "selected_primitive_id": selection.selected_primitive_id,
                "memory_label": "directional_residual_lift_belief_smoke_no_improvement_claim",
                "belief_before_local_lift_residual_m_s": belief_before["belief_local_lift_residual_m_s"],
                "belief_before_uncertainty": belief_before["belief_uncertainty"],
            }
        )
        row.update(archive_state_sample_row(state_sample))
        row.update(
            primitive_feature_row(
                primitive_feature_record(
                    state=state,
                    context=context,
                    primitive=primitive,
                    governor_mode=selection.governor_mode,
                    start_state_family=state_sample.start_state_family,
                    previous_primitive_status=state_sample.previous_primitive_status,
                    synthetic_time_since_launch_s=state_sample.synthetic_time_since_launch_s,
                )
            )
        )
        training_rows.append(row)
        episode_rows.append(row)
        selector_row = primitive_selection_row(selection)
        selector_row.update(archive_state_sample_row(state_sample))
        selector_rows.append(selector_row)
        belief = update_directional_residual_lift_belief(belief, _directional_observation_from_row(row))
        belief_after = _query_directional_belief(state, belief)
        row["belief_after_local_lift_residual_m_s"] = belief_after["belief_local_lift_residual_m_s"]
        row["belief_after_uncertainty"] = belief_after["belief_uncertainty"]
        belief_rows.append(belief_snapshot_row(belief, label=f"episode_{episode_index:03d}"))

    return _write_outputs(
        config=config,
        run_root=run_root,
        episode_rows=episode_rows,
        selector_rows=selector_rows,
        belief_rows=belief_rows,
    )


def _query_directional_belief(state: np.ndarray, belief) -> dict[str, object]:
    vector = np.asarray(state, dtype=float)
    return query_directional_residual_lift_features(
        belief,
        x_w_m=float(vector[0]),
        y_w_m=float(vector[1]),
        z_w_m=float(vector[2]),
        direction_rad=float(vector[5]) if vector.size > 5 else 0.0,
    )


def _directional_observation_from_row(row: dict[str, object]) -> DirectionalResidualObservation:
    return DirectionalResidualObservation(
        x_w_m=_float(row.get("initial_x_w", 0.0)),
        y_w_m=_float(row.get("initial_y_w", 0.0)),
        z_w_m=_float(row.get("initial_z_w", 0.0)),
        direction_rad=_float(row.get("initial_psi", 0.0)),
        lift_residual_m_s=_float(row.get("context_w_wing_mean_m_s", 0.0)),
        updraft_gain_residual_m=_float(
            row.get(
                "trajectory_integrated_updraft_gain_m",
                row.get("updraft_specific_energy_gain_proxy_m", max(_float(row.get("energy_residual_m", 0.0)), 0.0)),
            )
        ),
        dwell_residual_s=_float(row.get("lift_dwell_time_s", 0.0)),
        specific_energy_residual_m=_float(row.get("energy_residual_m", 0.0)),
    )


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _bootstrap_lqr_rows(
    *,
    state: np.ndarray,
    context,
    primitives,
    config: RolloutConfig,
    wind_field: object | None,
    episode_index: int,
    state_sample,
) -> list[dict[str, object]]:
    rows = []
    for primitive in primitives:
        controller = lqr_controller_for_primitive_id(primitive.primitive_id)
        evidence = simulate_primitive_rollout(
            rollout_id=f"bootstrap_{episode_index:03d}_{primitive.primitive_id}",
            episode_id=f"bootstrap_{episode_index:03d}",
            initial_state=state,
            context=context,
            primitive=primitive,
            config=config,
            wind_field=wind_field,
            controller=controller,
            controller_selection_status="nominal_unselected_smoke",
        )
        row = rollout_with_context_row(evidence, context)
        row.update(
            primitive_feature_row(
                primitive_feature_record(
                    state=state,
                    context=context,
                    primitive=primitive,
                    start_state_family=state_sample.start_state_family,
                    previous_primitive_status=state_sample.previous_primitive_status,
                    synthetic_time_since_launch_s=state_sample.synthetic_time_since_launch_s,
                )
            )
        )
        rows.append(row)
    return rows


# =============================================================================
# 3) Output Helpers
# =============================================================================
def _write_outputs(
    *,
    config: EpisodeSmokeConfig,
    run_root: Path,
    episode_rows: list[dict[str, object]],
    selector_rows: list[dict[str, object]],
    belief_rows: list[dict[str, object]],
) -> dict[str, object]:
    episode_log = run_root / "tables" / "episode_log.csv"
    selector_log = run_root / "tables" / "selector_log.csv"
    belief_log = run_root / "tables" / "belief_snapshots.csv"
    pd.DataFrame(episode_rows).to_csv(filesystem_path(episode_log), index=False)
    pd.DataFrame(selector_rows).to_csv(filesystem_path(selector_log), index=False)
    pd.DataFrame(belief_rows).to_csv(filesystem_path(belief_log), index=False)
    manifest = {
        "run_id": int(config.run_id),
        "episode_count": int(config.episode_count),
        "governor_mode": str(config.governor_mode),
        "rollout_backend": "model_backed_lqr",
        "evidence_role": "lqr_rollout_candidate",
        "memory_label_status": "directional_residual_lift_belief_smoke_only",
        "belief_memory_axes": ["x", "y", "z", "direction"],
        "claim_status": "simulation_only_episode_smoke_no_performance_claim",
        "blocked_claims": [
            "controller_performance",
            "repeated_launch_improvement",
            "mission_success",
            "hardware_readiness",
            "real_flight_transfer",
            "W2_W3_robustness",
        ],
    }
    manifest_path = run_root / "manifests" / "episode_smoke_manifest.json"
    filesystem_path(manifest_path).write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="ascii",
    )
    report_path = run_root / "reports" / "episode_smoke_report.md"
    filesystem_path(report_path).write_text(
        "\n".join(
            [
                "# Contextual Episode Smoke Report",
                "",
                f"- Run ID: `{config.run_id}`",
                f"- Episodes: `{config.episode_count}`",
                f"- Governor mode: `{config.governor_mode}`",
                "- Evidence role: `lqr_rollout_candidate`",
                "- Claim boundary: no performance, transfer, mission, robustness, or hardware-readiness claim.",
                "",
            ]
        ),
        encoding="ascii",
    )
    return {
        "run_root": run_root,
        "episode_log": episode_log,
        "selector_log": selector_log,
        "belief_log": belief_log,
        "manifest": manifest_path,
        "report": report_path,
    }


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    run_contextual_episode_smoke(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
