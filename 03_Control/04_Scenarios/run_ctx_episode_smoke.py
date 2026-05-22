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
from env_ctx import build_environment_context  # noqa: E402
from env_surrogate import (  # noqa: E402
    READY_STATUS,
    resolve_surrogate_binding,
    wind_field_for_binding,
)
from prim_cat import active_primitive_catalogue, primitive_by_id  # noqa: E402
from prim_model import fit_primitive_outcome_model  # noqa: E402
from prim_roll import (  # noqa: E402
    RolloutConfig,
    blocked_rollout_evidence,
    rollout_with_context_row,
    simulate_primitive_rollout,
)
from prim_select import primitive_selection_row, select_primitive  # noqa: E402
from run_ctx_archive import _metadata_for_row, _state_for_row  # noqa: E402


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

    for episode_index in range(int(config.episode_count)):
        row_index = int(config.seed + episode_index)
        w_layer = "W0" if episode_index % 2 == 0 else "W1"
        environment_mode = "dry_air" if w_layer == "W0" else "gaussian_single"
        state = _state_for_row(row_index)
        metadata = _metadata_for_row(
            w_layer=w_layer,
            environment_mode=environment_mode,
            seed=config.seed + episode_index,
        )
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
        if len(training_rows) < len(primitives):
            training_rows.extend(
                _bootstrap_feedback_rows(
                    state=state,
                    context=context,
                    primitives=primitives,
                    config=RolloutConfig(
                        W_layer=w_layer,
                        rollout_backend="model_backed_feedback",
                        wind_mode=binding.wind_mode,
                    ),
                    wind_field=wind,
                    episode_index=episode_index,
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
        primitive = primitive_by_id(selection.selected_primitive_id)
        rollout_config = RolloutConfig(
            W_layer=w_layer,
            rollout_backend="model_backed_feedback",
            wind_mode=binding.wind_mode,
        )
        rollout_id = f"episode_{config.run_id:03d}_{episode_index:03d}"
        if binding.surrogate_binding_status != READY_STATUS:
            evidence = blocked_rollout_evidence(
                rollout_id=rollout_id,
                episode_id=rollout_id,
                initial_state=state,
                context=context,
                primitive=primitive,
                config=rollout_config,
                failure_label="surrogate_binding_blocked",
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
            )
        row = rollout_with_context_row(evidence, context)
        row.update(
            {
                "episode_index": episode_index,
                "selector_governor_mode": selection.governor_mode,
                "selector_status": selection.decision_status,
                "selected_primitive_id": selection.selected_primitive_id,
                "memory_label": "placeholder_no_learning_claim",
            }
        )
        training_rows.append(row)
        episode_rows.append(row)
        selector_rows.append(primitive_selection_row(selection))

    return _write_outputs(
        config=config,
        run_root=run_root,
        episode_rows=episode_rows,
        selector_rows=selector_rows,
    )


def _bootstrap_feedback_rows(
    *,
    state: np.ndarray,
    context,
    primitives,
    config: RolloutConfig,
    wind_field: object | None,
    episode_index: int,
) -> list[dict[str, object]]:
    rows = []
    for primitive in primitives:
        evidence = simulate_primitive_rollout(
            rollout_id=f"bootstrap_{episode_index:03d}_{primitive.primitive_id}",
            episode_id=f"bootstrap_{episode_index:03d}",
            initial_state=state,
            context=context,
            primitive=primitive,
            config=config,
            wind_field=wind_field,
        )
        rows.append(rollout_with_context_row(evidence, context))
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
) -> dict[str, object]:
    episode_log = run_root / "tables" / "episode_log.csv"
    selector_log = run_root / "tables" / "selector_log.csv"
    pd.DataFrame(episode_rows).to_csv(filesystem_path(episode_log), index=False)
    pd.DataFrame(selector_rows).to_csv(filesystem_path(selector_log), index=False)
    manifest = {
        "run_id": int(config.run_id),
        "episode_count": int(config.episode_count),
        "governor_mode": str(config.governor_mode),
        "rollout_backend": "model_backed_feedback",
        "evidence_role": "feedback_rollout_candidate",
        "memory_label_status": "placeholder_only",
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
                "- Evidence role: `feedback_rollout_candidate`",
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
        "manifest": manifest_path,
        "report": report_path,
    }


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    run_contextual_episode_smoke(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
