from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flight_dynamics import adapt_glider
from glider import build_nausicaa_glider
from implementation_wrappers import (
    ImplementationCommandWrapper,
    ImplementationWrapperConfig,
)
from linearisation import STATE_INDEX, key_derivatives, linearise_trim
from primitives import (
    BankReversalPrimitive,
    FlightPrimitive,
    NominalGlidePrimitive,
    RecoveryPrimitive,
    build_primitive_context,
)
from simulator import SimulationConfig, simulate_primitive, write_result_log
from viability_governor import ViabilityGovernor
from wind_scenarios import WindScenario, fixed_wind_scenarios


OUT_DIR = Path(__file__).resolve().parent / "C_results" / "s4_lite"
LOG_DIR = OUT_DIR / "logs"


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    primitive: FlightPrimitive
    wind: WindScenario
    x0: np.ndarray
    wrapper_config: ImplementationWrapperConfig


def _trimmed_state_at_altitude(x_trim: np.ndarray, altitude_m: float) -> np.ndarray:
    x0 = np.asarray(x_trim, dtype=float).reshape(15).copy()
    x0[STATE_INDEX["z_w"]] = float(altitude_m)
    return x0


def _recovery_entry_state(x_trim: np.ndarray) -> np.ndarray:
    x0 = _trimmed_state_at_altitude(x_trim, altitude_m=8.0)
    x0[STATE_INDEX["phi"]] = np.deg2rad(18.0)
    x0[STATE_INDEX["theta"]] += np.deg2rad(4.0)
    x0[STATE_INDEX["p"]] = np.deg2rad(-3.0)
    x0[STATE_INDEX["q"]] = np.deg2rad(1.0)
    x0[STATE_INDEX["r"]] = np.deg2rad(2.0)
    return x0


def _scenario_specs(x_trim: np.ndarray) -> list[ScenarioSpec]:
    winds = fixed_wind_scenarios()
    direct = ImplementationWrapperConfig()
    delayed = ImplementationWrapperConfig(
        name="deterministic_surface_delay_proxy",
        extra_delay_s=(0.02, 0.00, 0.02),
    )
    x_base = _trimmed_state_at_altitude(x_trim, altitude_m=8.0)
    return [
        ScenarioSpec(
            name="nominal_glide_zero_wind",
            primitive=NominalGlidePrimitive(duration_s=5.0),
            wind=winds["none"],
            x0=x_base.copy(),
            wrapper_config=direct,
        ),
        ScenarioSpec(
            name="nominal_glide_crosswind",
            primitive=NominalGlidePrimitive(duration_s=5.0),
            wind=winds["crosswind"],
            x0=x_base.copy(),
            wrapper_config=direct,
        ),
        ScenarioSpec(
            name="bank_reversal_zero_wind",
            primitive=BankReversalPrimitive(duration_s=7.0),
            wind=winds["none"],
            x0=x_base.copy(),
            wrapper_config=direct,
        ),
        ScenarioSpec(
            name="bank_reversal_mild_updraft_proxy",
            primitive=BankReversalPrimitive(duration_s=7.0),
            wind=winds["mild_updraft_proxy"],
            x0=x_base.copy(),
            wrapper_config=direct,
        ),
        ScenarioSpec(
            name="recovery_perturbed_strong_updraft_proxy",
            primitive=RecoveryPrimitive(duration_s=5.5),
            wind=winds["strong_updraft_proxy"],
            x0=_recovery_entry_state(x_trim),
            wrapper_config=delayed,
        ),
        ScenarioSpec(
            name="invalid_low_altitude_rejection",
            primitive=BankReversalPrimitive(duration_s=7.0),
            wind=winds["none"],
            x0=_trimmed_state_at_altitude(x_trim, altitude_m=0.10),
            wrapper_config=direct,
        ),
    ]


def _write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_metric(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_report(
    path: Path,
    metrics_rows: list[dict[str, object]],
    entry_rows: list[dict[str, object]],
    derivative_signs: dict[str, float],
    wrapper_summaries: list[dict[str, object]],
) -> None:
    lines = [
        "# S4-Lite Validation Report",
        "",
        "Measured residual-GP data were not available to this runner. Updraft stress cases marked as proxy use a deterministic analytic residual proxy.",
        "",
        "## Linearisation Sign Snapshot",
        "",
        "| Check | Value | Status |",
        "|---|---:|---|",
    ]
    for name in ("l_delta_a", "m_delta_e", "n_delta_r"):
        value = derivative_signs[name]
        lines.append(f"| {name} > 0 | {value:.6e} | {'pass' if value > 0.0 else 'fail'} |")

    lines.extend(
        [
            "",
            "## Entry Conditions",
            "",
            "| Scenario | Primitive | Accepted | Reasons |",
            "|---|---|---|---|",
        ]
    )
    for row in entry_rows:
        lines.append(
            f"| {row['scenario']} | {row['primitive']} | {row['accepted']} | {row['reasons']} |"
        )

    lines.extend(
        [
            "",
            "## Scenario Metrics",
            "",
            "| Scenario | Primitive | Status | Duration s | Altitude loss m | Final speed m/s | Max bank deg | Max alpha deg | Log |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in metrics_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["scenario"]),
                    str(row["primitive"]),
                    str(row["status"]),
                    _format_metric(row["duration_completed_s"]),
                    _format_metric(row["altitude_loss_m"]),
                    _format_metric(row["final_speed_m_s"]),
                    _format_metric(row["max_abs_phi_deg"]),
                    _format_metric(row["max_abs_alpha_deg"]),
                    str(row["log_path"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Implementation Wrappers",
            "",
            "| Name | Limit deg | Deadband deg | Quantization deg | Extra delay s |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    seen = set()
    for summary in wrapper_summaries:
        name = str(summary["name"])
        if name in seen:
            continue
        seen.add(name)
        lines.append(
            f"| {name} | {summary['command_limit_deg']} | {summary['deadband_deg']} | "
            f"{summary['quantization_deg']} | {summary['extra_delay_s']} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    aircraft = adapt_glider(build_nausicaa_glider())
    linear_model = linearise_trim(aircraft=aircraft)
    context = build_primitive_context(
        x_trim=linear_model.x_trim,
        u_trim=linear_model.u_trim,
        min_entry_altitude_m=0.75,
    )
    governor = ViabilityGovernor()
    sim_config = SimulationConfig()

    metrics_rows: list[dict[str, object]] = []
    entry_rows: list[dict[str, object]] = []
    wrapper_summaries: list[dict[str, object]] = []

    for spec in _scenario_specs(linear_model.x_trim):
        decision = governor.evaluate(
            scenario_name=spec.name,
            primitive=spec.primitive,
            x0=spec.x0,
            context=context,
        )
        entry_rows.append(
            {
                "scenario": spec.name,
                "primitive": spec.primitive.name,
                "accepted": decision.accepted,
                "reasons": "; ".join(decision.reasons),
            }
        )
        wrapper = ImplementationCommandWrapper(spec.wrapper_config)
        wrapper_summaries.append(wrapper.summary())
        if not decision.accepted:
            continue

        result = simulate_primitive(
            scenario_name=spec.name,
            primitive=spec.primitive,
            x0=spec.x0,
            context=context,
            aircraft=aircraft,
            wind_model=spec.wind.wind_model,
            wind_mode=spec.wind.wind_mode,
            wrapper=wrapper,
            config=sim_config,
        )
        log_path = LOG_DIR / f"{spec.name}.csv"
        write_result_log(result, log_path)
        metrics = dict(result.metrics)
        metrics["wind_label"] = spec.wind.label
        metrics["wind_mode"] = spec.wind.wind_mode
        metrics["residual_source"] = spec.wind.residual_source
        metrics["wrapper_name"] = spec.wrapper_config.name
        metrics["log_path"] = str(log_path)
        metrics_rows.append(metrics)

    metrics_path = OUT_DIR / "scenario_metrics.csv"
    rejection_path = OUT_DIR / "governor_rejections.csv"
    report_path = OUT_DIR / "validation_report.md"
    _write_metrics(metrics_path, metrics_rows)
    governor.write_rejection_log(rejection_path)
    _write_report(
        path=report_path,
        metrics_rows=metrics_rows,
        entry_rows=entry_rows,
        derivative_signs=key_derivatives(linear_model),
        wrapper_summaries=wrapper_summaries,
    )

    completed_count = sum(1 for row in metrics_rows if row["status"] == "completed")
    rejected_count = len(governor.rejection_rows)
    if completed_count < 5:
        raise SystemExit("fewer than five accepted scenarios completed")
    if rejected_count < 1:
        raise SystemExit("governor did not reject any primitive")
    if any(row["status"] != "completed" for row in metrics_rows):
        raise SystemExit("one or more accepted scenarios terminated early")

    print("s4-lite validation complete")
    print(f"metrics: {metrics_path}")
    print(f"governor_rejections: {rejection_path}")
    print(f"report: {report_path}")
    print(f"accepted_scenarios_completed: {completed_count}")
    print(f"rejected_primitives: {rejected_count}")


if __name__ == "__main__":
    main()
