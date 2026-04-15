from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


INNER_LOOP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INNER_LOOP_DIR))

from glider import build_nausicaa_glider


# =============================================================================
# SECTION MAP
# =============================================================================
# 1) Mass-property dataclasses
# 2) Inertia helpers
# 3) Surface mass elements
# 4) Baseline estimate case
# 5) Output writer
# =============================================================================


# =============================================================================
# 1) Mass-Property Dataclasses
# =============================================================================
@dataclass(frozen=True)
class MassElement:
    mass_kg: float
    r_cg_b: np.ndarray
    inertia_b: np.ndarray


@dataclass(frozen=True)
class MassEstimate:
    mass_kg: float
    r_cg_b: np.ndarray
    inertia_b: np.ndarray


# =============================================================================
# 2) Inertia Helpers
# =============================================================================
def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def _frame_from_axes(
    x_axis_b: np.ndarray,
    y_axis_b: np.ndarray,
) -> np.ndarray:
    x_dir_b = _unit(np.asarray(x_axis_b, dtype=float).reshape(3))
    y_dir_b = _unit(np.asarray(y_axis_b, dtype=float).reshape(3))
    z_dir_b = _unit(np.cross(x_dir_b, y_dir_b))
    y_dir_b = _unit(np.cross(z_dir_b, x_dir_b))
    return np.column_stack((x_dir_b, y_dir_b, z_dir_b))


def point_mass(
    mass_kg: float,
    r_cg_b: np.ndarray,
) -> MassElement:
    return MassElement(
        mass_kg=float(mass_kg),
        r_cg_b=np.asarray(r_cg_b, dtype=float).reshape(3),
        inertia_b=np.zeros((3, 3)),
    )


def rectangular_prism(
    mass_kg: float,
    size_x_m: float,
    size_y_m: float,
    size_z_m: float,
    r_cg_b: np.ndarray,
    c_bl: np.ndarray,
) -> MassElement:
    i_local = (float(mass_kg) / 12.0) * np.diag(
        [
            size_y_m ** 2 + size_z_m ** 2,
            size_x_m ** 2 + size_z_m ** 2,
            size_x_m ** 2 + size_y_m ** 2,
        ]
    )
    return MassElement(
        mass_kg=float(mass_kg),
        r_cg_b=np.asarray(r_cg_b, dtype=float).reshape(3),
        inertia_b=c_bl @ i_local @ c_bl.T,
    )


def x_axis_tube(
    mass_kg: float,
    length_m: float,
    radius_m: float,
    r_cg_b: np.ndarray,
) -> MassElement:
    r_sq_m2 = radius_m ** 2
    inertia_b = np.diag(
        [
            0.5 * mass_kg * r_sq_m2,
            (mass_kg / 12.0) * (length_m ** 2 + 3.0 * r_sq_m2),
            (mass_kg / 12.0) * (length_m ** 2 + 3.0 * r_sq_m2),
        ]
    )
    return MassElement(
        mass_kg=float(mass_kg),
        r_cg_b=np.asarray(r_cg_b, dtype=float).reshape(3),
        inertia_b=inertia_b,
    )


def combine_mass_elements(
    elements: list[MassElement],
) -> MassEstimate:
    mass_kg = sum(element.mass_kg for element in elements)
    r_cg_b = sum(element.mass_kg * element.r_cg_b for element in elements) / mass_kg
    inertia_b = np.zeros((3, 3))
    for element in elements:
        dr_b = element.r_cg_b - r_cg_b
        inertia_b += element.inertia_b + element.mass_kg * (
            np.dot(dr_b, dr_b) * np.eye(3) - np.outer(dr_b, dr_b)
        )
    return MassEstimate(
        mass_kg=mass_kg,
        r_cg_b=r_cg_b,
        inertia_b=inertia_b,
    )


# =============================================================================
# 3) Surface Mass Elements
# =============================================================================
def horizontal_surface_elements(
    root_le_b: np.ndarray,
    chord_m: float,
    span_m: float,
    dihedral_deg: float,
    mass_kg: float,
    thickness_m: float,
) -> list[MassElement]:
    half_span_m = 0.5 * span_m
    quarter_span_m = 0.25 * span_m
    dihedral_rad = np.deg2rad(dihedral_deg)
    half_mass_kg = 0.5 * mass_kg
    elements: list[MassElement] = []
    for side_sign in (1.0, -1.0):
        span_axis_b = np.array(
            [0.0, side_sign * np.cos(dihedral_rad), -np.sin(dihedral_rad)]
        )
        c_bl = _frame_from_axes(np.array([1.0, 0.0, 0.0]), span_axis_b)
        r_cg_b = np.array(
            [
                root_le_b[0] + 0.5 * chord_m,
                side_sign * quarter_span_m * np.cos(dihedral_rad),
                root_le_b[2] - quarter_span_m * np.sin(dihedral_rad),
            ]
        )
        elements.append(
            rectangular_prism(
                mass_kg=half_mass_kg,
                size_x_m=chord_m,
                size_y_m=half_span_m,
                size_z_m=thickness_m,
                r_cg_b=r_cg_b,
                c_bl=c_bl,
            )
        )
    return elements


def vertical_surface_element(
    root_le_b: np.ndarray,
    chord_m: float,
    height_m: float,
    mass_kg: float,
    thickness_m: float,
) -> MassElement:
    c_bl = _frame_from_axes(
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, -1.0]),
    )
    r_cg_b = np.array(
        [
            root_le_b[0] + 0.5 * chord_m,
            0.0,
            root_le_b[2] - 0.5 * height_m,
        ]
    )
    return rectangular_prism(
        mass_kg=mass_kg,
        size_x_m=chord_m,
        size_y_m=height_m,
        size_z_m=thickness_m,
        r_cg_b=r_cg_b,
        c_bl=c_bl,
    )


# =============================================================================
# 4) Baseline Estimate Case
# =============================================================================
def build_baseline_estimate() -> MassEstimate:
    glider = build_nausicaa_glider()
    wing, htail, vtail = glider.surfaces
    elements = (
        horizontal_surface_elements(
            root_le_b=wing.root_le_b,
            chord_m=wing.chord_m,
            span_m=wing.span_m,
            dihedral_deg=wing.dihedral_deg,
            mass_kg=0.0250,
            thickness_m=0.0060,
        )
        + horizontal_surface_elements(
            root_le_b=htail.root_le_b,
            chord_m=htail.chord_m,
            span_m=htail.span_m,
            dihedral_deg=0.0,
            mass_kg=0.0060,
            thickness_m=0.0030,
        )
        + [
            vertical_surface_element(
                root_le_b=vtail.root_le_b,
                chord_m=vtail.chord_m,
                height_m=vtail.span_m,
                mass_kg=0.0025,
                thickness_m=0.0030,
            ),
            x_axis_tube(
                mass_kg=0.0080,
                length_m=0.3500,
                radius_m=0.0030,
                r_cg_b=np.array([-0.0600, 0.0000, 0.0000]),
            ),
            point_mass(0.0270, np.array([0.0050, 0.0000, 0.0000])),
            point_mass(0.0170, np.array([0.0200, 0.0000, 0.0000])),
            point_mass(0.0015, np.array([0.0150, 0.0000, 0.0000])),
            point_mass(0.0015, np.array([0.0100, 0.0000, 0.0000])),
            point_mass(0.0025, np.array([0.0600, 0.2200, -0.0350])),
            point_mass(0.0025, np.array([0.0600, -0.2200, -0.0350])),
            point_mass(0.0025, np.array([-0.0100, 0.0000, 0.0000])),
            point_mass(0.0025, np.array([-0.0100, 0.0000, 0.0000])),
            point_mass(0.0150, np.array([-0.2550, 0.0000, 0.0100])),
            point_mass(0.0010, np.array([-0.1400, 0.0000, 0.0000])),
            point_mass(0.0030, np.array([-0.2350, 0.0000, 0.0000])),
        ]
    )
    return combine_mass_elements(elements)


# =============================================================================
# 5) Output Writer
# =============================================================================
def write_mass_properties_module(
    output_path: Path,
    estimate: MassEstimate,
) -> None:
    lines = [
        "from __future__ import annotations",
        "",
        "import numpy as np",
        "",
        "# Generated by build_mass_properties_estimate.py",
        f"MASS_KG = {estimate.mass_kg:.16f}",
        (
            "R_CG_B = np.array("
            f"{estimate.r_cg_b.tolist()}, dtype=float)"
        ),
        (
            "INERTIA_B = np.array("
            f"{estimate.inertia_b.tolist()}, dtype=float)"
        ),
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> None:
    estimate = build_baseline_estimate()
    output_path = Path(__file__).with_name("mass_properties_estimate.py")
    write_mass_properties_module(output_path, estimate)
    print("mass_kg", estimate.mass_kg)
    print("r_cg_b", estimate.r_cg_b)
    print("inertia_b")
    print(estimate.inertia_b)
    print("output", output_path)


if __name__ == "__main__":
    main()
