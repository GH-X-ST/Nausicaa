from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def _add_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    for rel in (
        "03_Control/02_Inner_Loop",
        "03_Control/03_Primitives",
        "03_Control/04_Scenarios",
    ):
        path = repo_root / rel
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return repo_root


REPO_ROOT = _add_paths()

from arena import ArenaConfig  # noqa: E402
from flight_dynamics import adapt_glider  # noqa: E402
from glider import build_nausicaa_glider  # noqa: E402
from governor import ViabilityGovernor  # noqa: E402
from latency import CommandToSurfaceLayer, LatencyEnvelope  # noqa: E402
from linearisation import linearise_trim  # noqa: E402
from primitive import build_primitive_context  # noqa: E402
from rollout import RolloutConfig, simulate_primitive, write_log  # noqa: E402
from scenarios import build_scenario  # noqa: E402


METRICS_DIR = REPO_ROOT / "03_Control" / "05_Results" / "metrics"
LOG_DIR = REPO_ROOT / "03_Control" / "05_Results" / "logs"


def run_scenario(scenario_id: str, seed: int) -> dict[str, object]:
    aircraft = adapt_glider(build_nausicaa_glider())
    linear_model = linearise_trim(aircraft=aircraft)
    context = build_primitive_context(
        x_trim=linear_model.x_trim,
        u_trim=linear_model.u_trim,
        min_entry_altitude_m=0.75,
    )
    arena = ArenaConfig()
    scenario = build_scenario(scenario_id, linear_model.x_trim, REPO_ROOT)
    governor = ViabilityGovernor(arena_config=arena)
    decision = governor.evaluate(
        scenario_id=scenario.scenario_id,
        primitive=scenario.primitive,
        x0=scenario.x0,
        context=context,
    )

    log_path = LOG_DIR / f"{scenario_id}_seed{seed}.csv"
    metrics_path = METRICS_DIR / f"{scenario_id}_seed{seed}.csv"
    rejection_path = METRICS_DIR / f"{scenario_id}_seed{seed}_governor_rejections.csv"
    if not decision.accepted:
        governor.write_rejection_log(rejection_path)
        row = {
            "scenario_id": scenario_id,
            "seed": int(seed),
            "wind_model": scenario.wind_model_name,
            "wind_mode": scenario.wind_mode,
            "latency_mode": scenario.latency_config.mode,
            "primitive_selected": scenario.primitive.name,
            "success": False,
            "termination_reason": "governor_rejected",
            "height_change_m": 0.0,
            "terminal_speed_m_s": 0.0,
            "max_alpha_deg": 0.0,
            "max_abs_phi_deg": 0.0,
            "min_wall_distance_m": 0.0,
            "inside_safe_volume": False,
            "saturation_fraction": 0.0,
            "tracking_error_rms": "",
            "governor_rejection_reason": "; ".join(decision.reasons),
            "log_path_relative": "",
        }
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
    )
    write_log(result, log_path)
    row = dict(result.metrics)
    _write_single_row(metrics_path, row)
    return row


def _write_single_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    row = run_scenario(args.scenario, args.seed)
    print("scenario complete")
    print(f"scenario_id: {row['scenario_id']}")
    print(f"success: {row['success']}")
    print(f"metrics: 03_Control/05_Results/metrics/{args.scenario}_seed{args.seed}.csv")


if __name__ == "__main__":
    main()
