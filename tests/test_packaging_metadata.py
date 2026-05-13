from __future__ import annotations

import sys
import zipfile
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

import nausicaa_build_backend


def test_requirements_are_pip_style() -> None:
    lines = [
        line.strip()
        for line in Path("requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    assert lines
    assert all(not line.startswith("[") for line in lines)
    assert all("=" not in line or ">=" in line or "==" in line or "<=" in line for line in lines)
    assert any(line.startswith("aerosandbox") for line in lines)
    assert any(line.startswith("casadi") for line in lines)


def test_pyproject_keeps_editable_install_metadata_minimal() -> None:
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert data["build-system"]["build-backend"] == "nausicaa_build_backend"
    assert data["build-system"]["backend-path"] == ["."]
    assert data["project"]["name"] == "nausicaa"
    assert "03_Control/03_Primitives" in data["tool"]["pytest"]["ini_options"]["pythonpath"]


def test_custom_editable_backend_builds_path_wheel(tmp_path: Path) -> None:
    wheel_name = nausicaa_build_backend.build_editable(str(tmp_path))
    wheel_path = tmp_path / wheel_name

    assert wheel_path.exists()
    with zipfile.ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())
        assert "nausicaa_editable.pth" in names
        assert "nausicaa-0.1.1.dist-info/METADATA" in names
        pth_text = wheel.read("nausicaa_editable.pth").decode("utf-8")
    assert "03_Control" in pth_text
