from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Import path setup
# 2) Runner constants
# 3) Scenario execution
# 4) CSV and selection helpers
# 5) CLI entry point
# =============================================================================

# =============================================================================
# 1) Import Path Setup
# =============================================================================
def _add_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    for rel in (
        "03_Control/02_Inner_Loop",
        "03_Control/03_Primitives",
        "03_Control/04_Scenarios",
    ):
        path = repo_root / rel
        if str(path) not in sys.path:
            # Script execution resolves modules from the repository layout, not installed packages.
            sys.path.insert(0, str(path))
    return repo_root


REPO_ROOT = _add_paths()

from arena import ArenaConfig  # noqa: E402
from flight_dynamics import adapt_glider  # noqa: E402
from glider import build_nausicaa_glider  # noqa: E402
from governor import ViabilityGovernor  # noqa: E402
from latency import (  # noqa: E402
    CommandToSurfaceLayer,
    LatencyEnvelope,
    half_response_s,
    latency_audit_fields,
    latency_range_label,
)
from linearisation import linearise_trim  # noqa: E402
from metrics import rejected_metrics  # noqa: E402
from primitive import build_primitive_context  # noqa: E402
from rollout import RolloutConfig, simulate_primitive, write_log  # noqa: E402
from scenarios import build_scenario  # noqa: E402


# =============================================================================
# 2) Runner Constants
# =============================================================================
# Default direct-run locations are nested under a numbered housekeeping folder.
DEFAULT_RUNNER_ROOT = (
    REPO_ROOT / "03_Control" / "05_Results" / "99_misc" / "01_direct_runner" / "000"
)


# =============================================================================
# 3) Scenario Execution
# =============================================================================
def _output_dirs(output_root: str | Path | None) -> tuple[Path, Path]:
    if output_root is None:
        return DEFAULT_RUNNER_ROOT / "metrics", DEFAULT_RUNNER_ROOT / "logs"
    # Test runs redirect logs and metrics without changing scenario semantics
    root = Path(output_root)
    return root / "metrics", root / "logs"


def run_scenario(
    scenario_id: str,
    seed: int,
    output_root: str | Path | None = None,
    file_stem: str | None = None,
) -> dict[str, object]:
    aircraft = adapt_glider(build_nausicaa_glider())
    linear_model = linearise_trim(aircraft=aircraft)
    context = build_primitive_context(
        x_trim=linear_model.x_trim,
        u_trim=linear_model.u_trim,
        min_entry_altitude_m=0.75,
    )
    arena = ArenaConfig()
    scenario = build_scenario(scenario_id, linear_model.x_trim, REPO_ROOT, seed=seed)
    governor = ViabilityGovernor(arena_config=arena)

    # Each scenario run writes one metrics row and an optional rollout log
    metrics_dir, log_dir = _output_dirs(output_root)
    stem = file_stem or f"{scenario_id}_seed{seed}"
    log_path = log_dir / f"{stem}.csv"
    metrics_path = metrics_dir / f"{stem}.csv"
    rejection_path = metrics_dir / f"{stem}_governor_rejections.csv"
    candidate_path = metrics_dir / f"{stem}_governor_candidates.csv"

    if scenario.candidate_primitives:
        # Candidate-primitives scenarios exercise rollout-based governor selection
        decision = governor.select_primitive(
            scenario_id=scenario.scenario_id,
            primitives=scenario.candidate_primitives,
            x0=scenario.x0,
            context=context,
            rollout_callable=lambda primitive: simulate_primitive(
                scenario_id=scenario.scenario_id,
                seed=seed,
                primitive=primitive,
                x0=scenario.x0,
                context=context,
                aircraft=aircraft,
                wind_model=scenario.wind_model,
                wind_model_name=scenario.wind_model_name,
                wind_mode=scenario.wind_mode,
                command_layer=CommandToSurfaceLayer(
                    config=scenario.latency_config,
                    envelope=LatencyEnvelope(),
                ),
                log_path=log_path,
                repo_root=REPO_ROOT,
                rollout_config=RolloutConfig(),
                arena_config=arena,
                wind_param_label=scenario.wind_param_label,
            ),
        )
        governor.write_candidate_table(candidate_path, decision.candidate_table)
        governor.write_rejection_log(rejection_path)
        selected = _selected_primitive(scenario.candidate_primitives, decision.selected_primitive_name)
        if selected is None:
            envelope = LatencyEnvelope()
            row = rejected_metrics(
                scenario_id=scenario_id,
                seed=seed,
                wind_model=scenario.wind_model_name,
                wind_mode=scenario.wind_mode,
                wind_param_label=scenario.wind_param_label,
                latency_mode=scenario.latency_config.mode,
                latency_s=half_response_s(scenario.latency_config, envelope),
                latency_range_s=latency_range_label(scenario.latency_config, envelope),
                **latency_audit_fields(scenario.latency_config, envelope),
                primitive_name=scenario.primitive.name,
                x0=scenario.x0,
                log_path=log_path,
                repo_root=REPO_ROOT,
                arena_config=arena,
                governor_rejection_reason=decision.fallback_reason or "; ".join(decision.reasons),
                candidate_count=len(decision.candidate_table),
                rejected_count=len(decision.candidate_table),
                primitive_metadata=_first_candidate_metadata(scenario.candidate_primitives),
            )
            row["candidate_table_path"] = _relative_output_path(candidate_path, output_root)
            _write_single_row(metrics_path, row)
            return row
        result = simulate_primitive(
            scenario_id=scenario.scenario_id,
            seed=seed,
            primitive=selected,
            x0=scenario.x0,
            context=context,
            aircraft=aircraft,
            wind_model=scenario.wind_model,
            wind_model_name=scenario.wind_model_name,
            wind_mode=scenario.wind_mode,
            command_layer=CommandToSurfaceLayer(
                config=scenario.latency_config,
                envelope=LatencyEnvelope(),
            ),
            log_path=log_path,
            repo_root=REPO_ROOT,
            rollout_config=RolloutConfig(),
            arena_config=arena,
            wind_param_label=scenario.wind_param_label,
            selected_primitive_name=decision.selected_primitive_name,
            governor_rejection_reason=decision.fallback_reason or "; ".join(decision.reasons),
            candidate_count=len(decision.candidate_table),
            rejected_count=sum(
                1 for candidate in decision.candidate_table if not candidate.predicted_safe
            ),
        )
        write_log(result, log_path)
        row = dict(result.metrics)
        row["candidate_table_path"] = _relative_output_path(candidate_path, output_root)
        _write_single_row(metrics_path, row)
        return row

    decision = governor.evaluate(
        scenario_id=scenario.scenario_id,
        primitive=scenario.primitive,
        x0=scenario.x0,
        context=context,
    )
    if not decision.accepted:
        governor.write_rejection_log(rejection_path)
        envelope = LatencyEnvelope()
        row = rejected_metrics(
            scenario_id=scenario_id,
            seed=seed,
            wind_model=scenario.wind_model_name,
            wind_mode=scenario.wind_mode,
            wind_param_label=scenario.wind_param_label,
            latency_mode=scenario.latency_config.mode,
            latency_s=half_response_s(scenario.latency_config, envelope),
            latency_range_s=latency_range_label(scenario.latency_config, envelope),
            **latency_audit_fields(scenario.latency_config, envelope),
            primitive_name=scenario.primitive.name,
            x0=scenario.x0,
            log_path=log_path,
            repo_root=REPO_ROOT,
            arena_config=arena,
            governor_rejection_reason="; ".join(decision.reasons),
            primitive_metadata=getattr(scenario.primitive, "metadata", None),
        )
        row["candidate_table_path"] = ""
        _write_single_row(metrics_path, row)
        return row

    command_layer = CommandToSurfaceLayer(
        config=scenario.latency_config,
        envelope=LatencyEnvelope(),
    )
    result = simulate_primitive(
        scenario_id=scenario.scenario_id,
        seed=seed,
        primitive=scenario.primitive,
        x0=scenario.x0,
        context=context,
        aircraft=aircraft,
        wind_model=scenario.wind_model,
        wind_model_name=scenario.wind_model_name,
        wind_mode=scenario.wind_mode,
        command_layer=command_layer,
        log_path=log_path,
        repo_root=REPO_ROOT,
        rollout_config=RolloutConfig(),
        arena_config=arena,
        wind_param_label=scenario.wind_param_label,
    )
    write_log(result, log_path)
    row = dict(result.metrics)
    row["candidate_table_path"] = ""
    _write_single_row(metrics_path, row)
    return row


# =============================================================================
# 4) CSV and Selection Helpers
# =============================================================================
# CSV writer uses the row's stable metrics schema for single-scenario result files.
def _write_single_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _selected_primitive(
    primitives: tuple[object, ...],
    selected_name: str | None,
) -> object | None:
    for primitive in primitives:
        if primitive.name == selected_name:
            return primitive
    return None


def _first_candidate_metadata(primitives: tuple[object, ...]) -> dict[str, object] | None:
    for primitive in primitives:
        metadata = getattr(primitive, "metadata", None)
        if metadata is not None:
            return dict(metadata)
    return None


def _relative_output_path(path: Path, output_root: str | Path | None) -> str:
    base = REPO_ROOT if output_root is None else Path(output_root)
    try:
        return str(path.resolve().relative_to(base.resolve())).replace("\\", "/")
    except ValueError:
        return path.name


# =============================================================================
# 5) CLI Entry Point
# =============================================================================
# CLI and tests delegate to run_scenario to avoid divergent scenario semantics.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    row = run_scenario(args.scenario, args.seed, output_root=args.output_root)
    print("scenario complete")
    print(f"scenario_id: {row['scenario_id']}")
    print(f"success: {row['success']}")
    metrics_dir, _log_dir = _output_dirs(args.output_root)
    print(f"metrics: {metrics_dir / f'{args.scenario}_seed{args.seed}.csv'}")


if __name__ == "__main__":
    main()
