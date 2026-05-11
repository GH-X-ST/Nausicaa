from __future__ import annotations

from pathlib import Path

import numpy as np

from linearisation import (
    LATERAL_INPUTS,
    LATERAL_STATES,
    LONGITUDINAL_INPUTS,
    LONGITUDINAL_STATES,
    STATE_INDEX,
    key_derivatives,
    linearise_trim,
    reduced_model,
)
from trim_solver import TrimTarget


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Output path
# 2) Markdown helpers
# 3) Linearisation report builder
# 4) CLI entry point
# =============================================================================

# =============================================================================
# 1) Output Path
# =============================================================================
OUT_PATH = Path(__file__).resolve().parent / "C_results" / "linearisation_audit.md"


# =============================================================================
# 2) Markdown Helpers
# =============================================================================
def _row(name: str, value: object) -> str:
    return f"| {name} | {value} |"


def _shape_text(value: np.ndarray) -> str:
    return str(tuple(int(v) for v in value.shape))


# =============================================================================
# 3) Linearisation Report Builder
# =============================================================================
def _markdown_report() -> tuple[str, float, bool]:
    target = TrimTarget(speed_m_s=6.5)
    model = linearise_trim(target=target)
    # Reduced views preserve the canonical state/input ordering defined in linearisation.py.
    a_lon, b_lon = reduced_model(
        model,
        LONGITUDINAL_STATES,
        LONGITUDINAL_INPUTS,
    )
    a_lat, b_lat = reduced_model(
        model,
        LATERAL_STATES,
        LATERAL_INPUTS,
    )
    derivatives = key_derivatives(model)
    # Position rates are excluded because trim enforces dynamic equilibrium, not zero motion.
    residual = float(np.max(np.abs(model.f_trim[3:])))

    trim_rows = [
        ("V_trim_m_s", target.speed_m_s),
        ("alpha_rad", np.arctan2(model.x_trim[STATE_INDEX["w"]], model.x_trim[STATE_INDEX["u"]])),
        ("theta_rad", model.x_trim[STATE_INDEX["theta"]]),
        (
            "gamma_rad",
            np.arctan2(
                model.f_trim[2],
                max(np.linalg.norm(model.f_trim[:2]), 1e-12),
            ),
        ),
        ("sink_rate_m_s", -model.f_trim[2]),
        ("delta_a_rad", model.x_trim[STATE_INDEX["delta_a"]]),
        ("delta_e_rad", model.x_trim[STATE_INDEX["delta_e"]]),
        ("delta_r_rad", model.x_trim[STATE_INDEX["delta_r"]]),
    ]
    shape_rows = [
        ("A shape", _shape_text(model.a)),
        ("B_cmd shape", _shape_text(model.b)),
        ("A_longitudinal shape", _shape_text(a_lon)),
        ("B_longitudinal shape", _shape_text(b_lon)),
        ("A_lateral shape", _shape_text(a_lat)),
        ("B_lateral shape", _shape_text(b_lat)),
    ]
    sign_tests = [
        # Positive command-effectiveness signs are the compact regression audit for conventions.
        ("l_delta_a > 0", derivatives["l_delta_a"] > 0.0, derivatives["l_delta_a"]),
        ("m_delta_e > 0", derivatives["m_delta_e"] > 0.0, derivatives["m_delta_e"]),
        ("n_delta_r > 0", derivatives["n_delta_r"] > 0.0, derivatives["n_delta_r"]),
        ("delta_a_cmd > 0", derivatives["delta_a_cmd"] > 0.0, derivatives["delta_a_cmd"]),
        ("delta_e_cmd > 0", derivatives["delta_e_cmd"] > 0.0, derivatives["delta_e_cmd"]),
        ("delta_r_cmd > 0", derivatives["delta_r_cmd"] > 0.0, derivatives["delta_r_cmd"]),
    ]
    all_signs_pass = all(passed for _name, passed, _value in sign_tests)

    lines = [
        "# Linearisation Audit",
        "",
        "The command Jacobian reflects actuator lag. Aerodynamic control effectiveness appears in the full-state columns for actual surface deflections.",
        "",
        "## Trim Summary",
        "",
        "| Field | Value |",
        "|---|---:|",
    ]
    lines.extend(_row(name, f"{float(value):.6e}") for name, value in trim_rows)
    lines.extend(
        [
            "",
            "## Matrix Shapes",
            "",
            "| Field | Value |",
            "|---|---:|",
        ]
    )
    lines.extend(_row(name, value) for name, value in shape_rows)
    lines.extend(
        [
            "",
            "## Dynamic Residual",
            "",
            f"Maximum residual excluding position rates: `{residual:.6e}`",
            "",
            "## Reduced Model Dimensions",
            "",
            "| Model | A shape | B shape |",
            "|---|---:|---:|",
            f"| Longitudinal | {_shape_text(a_lon)} | {_shape_text(b_lon)} |",
            f"| Lateral-directional | {_shape_text(a_lat)} | {_shape_text(b_lat)} |",
            "",
            "## Key Derivatives",
            "",
            "| Derivative | Value |",
            "|---|---:|",
        ]
    )
    lines.extend(
        _row(name, f"{value:.6e}") for name, value in derivatives.items()
    )
    lines.extend(
        [
            "",
            "## Sign Audit",
            "",
            "| Check | Value | Status |",
            "|---|---:|---|",
        ]
    )
    for name, passed, value in sign_tests:
        lines.append(f"| {name} | {value:.6e} | {'pass' if passed else 'fail'} |")
    return "\n".join(lines) + "\n", residual, all_signs_pass


# =============================================================================
# 4) CLI Entry Point
# =============================================================================
def main() -> None:
    report, residual, all_signs_pass = _markdown_report()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # The report is a reproducible artifact used by project-plan validation commands.
    OUT_PATH.write_text(report, encoding="utf-8")
    print("linearisation audit complete")
    print(f"report: {OUT_PATH}")
    print(f"max_dynamic_residual: {residual:.6e}")
    if not all_signs_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
